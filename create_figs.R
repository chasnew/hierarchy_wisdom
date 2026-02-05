library(tidyverse)
library(cowplot)
library(scales)

wd_result_path <- '/Users/chanuwasaswamenakul/Documents/workspace/hierarchy_wisdom/results/'
box_path <- '/Users/chanuwasaswamenakul/Library/CloudStorage/Box-Box'


# Samples of initial beta distribution
# c("Beta(1.9,0.1)", "Beta(0.1,1.9)", "Beta(1,1)", "Beta(2,1)", "Beta(1,2)", "Beta(2,2)", "Beta(0.75,0.75)")
init_cond_labels <- c("Beta(1.9,0.1)", "Beta(0.1,1.9)", "Beta(1,1)")
init_hist_list <- list()

for (i in 1:length(init_cond_labels)) {
  label <- init_cond_labels[i]
  str_param <- gsub("[\\(\\)]", "", regmatches(label, gregexpr("\\(.*?\\)", label))[[1]])
  param <- str_split(str_param, ',')
  
  data <- rbeta(10000, as.numeric(param[[1]][1]), as.numeric(param[[1]][2]))
  dist_plot <- tibble(x = data) %>% 
    ggplot(aes(x = x)) +
    geom_histogram(bins = 50) +
    scale_x_continuous(name = "influence",
                       limits = c(-0.05,1.05),
                       expand = c(0,0)) +
    scale_y_continuous(expand = c(0,0)) +
    # ggtitle(label) +
    theme_classic() +
    theme(plot.title = element_text(size = 16, hjust = 0.5),
          axis.title.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank())
  
  init_hist_list[[i]] <- dist_plot
}

plot_grid(init_hist_list[[1]], init_hist_list[[2]], init_hist_list[[3]], init_hist_list[[4]],
          init_hist_list[[5]], init_hist_list[[6]], init_hist_list[[7]], ncol = 4)

ggsave(file.path(wd_result_path,
                 "init_distributions.jpg"),
       height = 6, width = 12)



# End state of influence distribution

init_cond_list <- c("most_leaders", "most_followers", "uniform", "a2b1", "a1b2", "a2b2", "a0.75b0.75")

inf_hist_list <- list()
row_title_list <- list()

ct <- 3
criterion <- "sd"

for (i in 1:length(init_cond_list)) {
  
  init_cond <- init_cond_list[i]
  print(paste(i, init_cond))
  
  for (sim in 1:5) {
    
    result_path <- file.path(box_path, 'HierarchyWisdom', 'results',
                             paste0('endpool_ct', ct, '_', criterion, '_', init_cond,
                                    '_alpha_sim', sim, '.csv'))
    
    final_state <- read_csv(result_path, col_names = FALSE)
    
    inf_hist <- final_state %>% 
      ggplot(aes(x=X1)) +
      geom_histogram(bins = 50) +
      scale_x_continuous(name = "influence", limits = c(0,1)) +
      theme_classic() +
      theme(axis.title.y=element_blank(),
            axis.text.y=element_blank(),
            axis.ticks.y=element_blank())
    
    ind <- (5*(i-1)) + sim
    inf_hist_list[[ind]] <- inf_hist
  }
  
  row_title <- ggdraw() + 
    draw_label(
      init_cond,
      fontface = 'bold',
      size = 10,
      y = 0.5,
      vjust = 0.5,
      angle = 90
    )
  
  row_title_list[[i]] <- row_title
}

plot_grid(row_title_list[[1]], inf_hist_list[[1]], inf_hist_list[[2]], inf_hist_list[[3]], inf_hist_list[[4]], inf_hist_list[[5]],
          row_title_list[[2]], inf_hist_list[[6]], inf_hist_list[[7]], inf_hist_list[[8]], inf_hist_list[[9]], inf_hist_list[[10]],
          row_title_list[[3]], inf_hist_list[[11]], inf_hist_list[[12]], inf_hist_list[[13]], inf_hist_list[[14]], inf_hist_list[[15]],
          row_title_list[[4]], inf_hist_list[[16]], inf_hist_list[[17]], inf_hist_list[[18]], inf_hist_list[[19]], inf_hist_list[[20]],
          row_title_list[[5]], inf_hist_list[[21]], inf_hist_list[[22]], inf_hist_list[[23]], inf_hist_list[[24]], inf_hist_list[[25]],
          row_title_list[[6]], inf_hist_list[[26]], inf_hist_list[[27]], inf_hist_list[[28]], inf_hist_list[[29]], inf_hist_list[[30]],
          row_title_list[[7]], inf_hist_list[[31]], inf_hist_list[[32]], inf_hist_list[[33]], inf_hist_list[[34]], inf_hist_list[[35]],
          ncol = 6, rel_widths = c(0.1, 1, 1, 1, 1, 1))

ggsave(file.path(wd_result_path,
                 paste0("end_pool_ct", ct, "_alpha_all.jpg")),
       height = 8, width = 12)






# Decision time with discrete leaders (not needed because of focus on group size)

# opf_result_path <- file.path(box_path, 'HierarchyWisdom', 'results', 'replicated_opf_results.csv')
# opf_result_df <- read_csv(opf_result_path)
# 
# # figure 1a
# sub_nleads <- c(0, 1, 10)
# sub_df <- opf_result_df %>% 
#   filter(nlead %in% sub_nleads)
# 
# # aggregate sub_df to calculate sd and average values of each N
# agg_df <- sub_df %>% 
#   group_by(N, nlead) %>% 
#   summarize(avg_t = mean(n_event), sd_t = sd(n_event))
# 
# decis_time_plot <- sub_df %>% 
#   ggplot(aes(x = N, group = nlead, color = as.factor(nlead))) +
#   geom_point(aes(y = n_event)) +
#   geom_ribbon(data = agg_df, aes(ymin = avg_t - (2*sd_t),
#                                  ymax = avg_t + (2*sd_t)),
#               alpha = .3) +
#   geom_line(data = agg_df, aes(y = avg_t), linewidth = 1.5) +
#   scale_y_continuous("Decision time") +
#   scale_x_continuous("group size") +
#   guides(color = guide_legend(title = "# of leaders")) +
#   theme_classic() +
#   theme(axis.title=element_text(size=16, face="bold"),
#         axis.text=element_text(size=12),
#         legend.text = element_text(size=12))


# Decision time heatmaps
lim_speakers <- 1
criterion_list <- c('sd', 'prop')

heatmap_list <- list()

beta_sample_locs <- data.frame(a = c(0.1, 1, 1.9),
                               b = c(1.9, 1, 0.1))

annotate_df <- data.frame(a = c(0.4, 1.2, 1.7),
                          b = c(1.9, 1.1, 0.2),
                          text = c("Beta(0.1,1.9)", "Beta(1,1)", "Beta(1.9,0.1)"))

# generate heatmap for 2 different decision rules
for (i in 1:2) {
  criterion <- criterion_list[i]
  
  beta_decis_path <- file.path(box_path, 'HierarchyWisdom', 'results',
                               paste0('opfsp', lim_speakers, '_', criterion,
                                      '_beta_decision_time.csv'))
  
  beta_decis_df <- read_csv(beta_decis_path)
  
  # low: #56B4E9, high: #E69F00
  
  heatmap_plot <- beta_decis_df %>% ggplot(aes(x=a, y=b)) + 
    geom_tile(aes(fill=avg_n_event)) +
    # geom_contour(aes(z=value),
    #              color="black",
    #              breaks = c(0.4, 1, 1.8, 2.4)) +
    geom_point(data = beta_sample_locs, color = "#212f3c", size = 4) +
    geom_text(data = annotate_df, aes(x=a, y=b, label=text),
              size=5 , fontface="bold") +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                         limits = c(0, 4600), midpoint = 2300) +
    geom_hline(yintercept = 1) +
    geom_vline(xintercept = 1) +
    scale_x_continuous(expression(a), limits = c(0.06,2.03), expand=c(0,0)) +
    scale_y_continuous(expression(b), limits = c(0.06,2.03), expand=c(0,0)) +
    guides(fill = guide_colourbar(title = "Decision time")) +
    theme_classic() +
    theme(panel.background = element_blank(),
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          axis.title=element_text(size=18,face="bold"),
          axis.text=element_text(size=14),
          legend.key.width  = unit(1, "lines"),
          legend.key.height = unit(2, "lines"),
          axis.line = element_line(colour = "black"),
          panel.border = element_rect(colour = "black",
                                      fill=NA, linewidth=3))
  
  heatmap_list[[i]] <- heatmap_plot
}

consensus_title <- ggdraw() + 
  draw_label(
    "Consensus Rule",
    fontface = 'bold',
    size = 16,
    x = 0.5,
    vjust = 0.5,
    angle = 0
  )

prop_title <- ggdraw() + 
  draw_label(
    "Proportional Voting",
    fontface = 'bold',
    size = 16,
    x = 0.5,
    vjust = 0.5,
    angle = 0
  )

init_cond_alt_labels <- c("Beta(1.9,0.1)", "Beta(0.1,1.9)", "Beta(1,1)")

beta_dist_titles <- list()
for (i in 1:length(init_cond_labels)) {
  label <- init_cond_alt_labels[i]
  beta_dist_titles[[i]] <- ggdraw() + 
    draw_label(
      label,
      size = 16,
      x = 0.5,
      vjust = 0.5,
      angle = 0
    )
}

tmp <- get_legend(heatmap_list[[1]])

heatmap_list[[1]] <- heatmap_list[[1]] + theme(legend.position = "none")
heatmap_list[[2]] <- heatmap_list[[2]] + theme(legend.position = "none")


heatmaps <- plot_grid(consensus_title, prop_title,
                      heatmap_list[[1]], heatmap_list[[2]],
                      ncol = 2, labels = c("a", "b"), label_size = 20,
                      rel_widths = c(1, 1, 0.2), rel_heights = c(0.1, 1))


beta_sample_grid <- plot_grid(
  NA, beta_dist_titles[[2]], beta_dist_titles[[3]], beta_dist_titles[[1]],
  NA, init_hist_list[[2]], init_hist_list[[3]], init_hist_list[[1]], 
  labels = c(NA, "c", "d", "e"), label_size = 20,
  ncol = 4, rel_widths = c(0.18, 1, 1, 1), rel_heights = c(0.3, 1)
)

plot_grid(heatmaps, tmp,
          beta_sample_grid, NA,
          ncol = 2, rel_widths = c(1, 0.1), rel_heights = c(1, 0.3))

ggsave(file.path(wd_result_path,
                 paste0("decision_time_plots.jpg")),
       height = 6, width = 12)






# Start and End state of influence distributions (samples)

ct_list <- c(3, -3)
init_cond_list <- c("most_leaders", "uniform", "most_followers")
init_cond_labels <- c("Start with\nmost leaders", "Start with\nuniform", "Start with\nmost followers")

inf_hist_list <- list()
row_title_list <- list()
col_title_list <- list()

criterion <- "sd"

# locations and labels for annotation of beta distributions
start_x_locs <- 0.5
start_y_locs <- c(4000, 270, 4000)
start_labels <- c("Beta(0.1,1.9)", "Beta(1,1)", "Beta(1.9,0.1)")

end_x_locs <- c(0.7, 0.3)
end_y_locs <- c(350, 375, 375, 375, 400, 375)
end_labels <- c("Beta(1.15, 1.7)", "Beta(2.24, 1.0)", "Beta(1.02, 1.79)",
                "Beta(1.11, 0.79)", "Beta(1.48, 4.22)", "Beta(1.03, 0.79)")

for (i in 1:length(init_cond_list)) {
  
  init_cond <- init_cond_list[i]
  print(paste(init_cond))
  
  for (j in 1:length(ct_list)) {
    
    ct <- ct_list[j]
    
    result_path <- file.path(box_path, 'HierarchyWisdom', 'results',
                             paste0('sliced_influence_ct', ct, '_', criterion, '_', init_cond,
                                    '_sim5.csv'))
    
    sliced_states <- read_csv(result_path)
    
    # geom_hist unintentionally remove small zero values (as non-finite values)
    # uniform distribution needs extra space for annotation
    if (i == 2) { # uniform histograms have lower height
      start_hist <- sliced_states %>% 
        ggplot(aes(x=first_infl)) +
        geom_bar(width = 1) +
        annotate("text", x=start_x_locs, y=start_y_locs[i], label=start_labels[i], size=4) +
        scale_x_binned(breaks = seq(0,1,0.02), expand = c(0.01,0.01)) +
        scale_y_continuous(init_cond_labels[i], limits = c(0, 300), expand = c(0,0)) +
        theme_classic() +
        theme(axis.title=element_text(size=12),
              axis.title.x=element_blank(),
              axis.text=element_blank(),
              axis.ticks=element_blank())
    } else if (i == 3) { # bottom row keep x axis title
      start_hist <- sliced_states %>% 
        ggplot(aes(x=first_infl)) +
        geom_bar(width = 1) +
        annotate("text", x=start_x_locs, y=start_y_locs[i], label=start_labels[i], size=4) +
        scale_x_binned("influence", breaks = seq(0,1,0.02), expand = c(0.01,0.01)) +
        scale_y_continuous(init_cond_labels[i], expand = c(0,0)) +
        theme_classic() +
        theme(axis.title=element_text(size=12),
              axis.text=element_blank(),
              axis.ticks=element_blank())
    } else {
      start_hist <- sliced_states %>% 
        ggplot(aes(x=first_infl)) +
        geom_bar(width = 1) +
        annotate("text", x=start_x_locs, y=start_y_locs[i], label=start_labels[i], size=4) +
        scale_x_binned(breaks = seq(0,1,0.02), expand = c(0.01,0.01)) +
        scale_y_continuous(init_cond_labels[i], expand = c(0,0)) +
        theme_classic() +
        theme(axis.title=element_text(size=12),
              axis.title.x=element_blank(),
              axis.text=element_blank(),
              axis.ticks=element_blank())
    }
    
    end_ind <- 2*i + j - 2
    
    if (i == 3) { # bottom row keep x axis title
      end_hist <- sliced_states %>% 
        ggplot(aes(x=last_infl)) +
        geom_bar(width = 1) +
        annotate("text", x=end_x_locs[j], y=end_y_locs[end_ind], label=end_labels[end_ind], size=4) +
        scale_x_binned("influence", breaks = seq(0,1,0.02), expand = c(0.01,0.01)) +
        scale_y_continuous(expand = c(0,0)) +
        theme_classic() +
        theme(axis.title.y=element_blank(),
              axis.text=element_blank(),
              axis.ticks=element_blank())
    } else {
      end_hist <- sliced_states %>% 
        ggplot(aes(x=last_infl)) +
        geom_bar(width = 1) +
        annotate("text", x=end_x_locs[j], y=end_y_locs[end_ind], label=end_labels[end_ind], size=4) +
        scale_x_binned(breaks = seq(0,1,0.02), expand = c(0.01,0.01)) +
        scale_y_continuous(expand = c(0,0)) +
        theme_classic() +
        theme(axis.title=element_blank(),
              axis.text=element_blank(),
              axis.ticks=element_blank())
    }
    
    
    
    # each row has 4 plots
    # each init_cond x C has 2 plots
    ind <- (4*(i-1)) + (2*(j-1) + 1)
    
    if ((ind %% 4) != 1) {
      start_hist <- start_hist + theme(axis.title.y=element_blank())
    }
    
    inf_hist_list[[ind]] <- start_hist
    inf_hist_list[[ind+1]] <- end_hist
    
  }
}

start_col <- ggdraw() + 
  draw_label(
    "start",
    size = 14,
    x = 0.5,
    hjust = 0.5,
    angle = 0
  )

end_col <- ggdraw() + 
  draw_label(
    "end",
    size = 14,
    x = 0.5,
    hjust = 0.5,
    angle = 0
  )

# most_lead_row <- ggdraw() + 
#   draw_label(
#     "Start with\nmostly leaders",
#     fontface = 'bold',
#     size = 12,
#     y = 0.5,
#     vjust = 0.5,
#     angle = 90
#   )
# 
# unif_row <- ggdraw() + 
#   draw_label(
#     "Start with\nuniform",
#     fontface = 'bold',
#     size = 12,
#     y = 0.5,
#     vjust = 0.5,
#     angle = 90
#   )
# 
# most_fllw_row <- ggdraw() + 
#   draw_label(
#     "Start with\nmostly followers",
#     fontface = 'bold',
#     size = 12,
#     y = 0.5,
#     vjust = 0.5,
#     angle = 90
#   )

# plot_grid(NA, start_col, end_col, start_col, end_col,
#           most_lead_row, inf_hist_list[[1]], inf_hist_list[[2]], inf_hist_list[[3]], inf_hist_list[[4]],
#           most_fllw_row, inf_hist_list[[5]], inf_hist_list[[6]], inf_hist_list[[7]], inf_hist_list[[8]],
#           ncol = 5, rel_widths = c(0.1, 1, 1, 1, 1), rel_heights = c(0.1, 1, 1))


plot_grid(start_col, end_col, start_col, end_col,
          inf_hist_list[[1]], inf_hist_list[[2]], inf_hist_list[[3]], inf_hist_list[[4]],
          inf_hist_list[[5]], inf_hist_list[[6]], inf_hist_list[[7]], inf_hist_list[[8]],
          inf_hist_list[[9]], inf_hist_list[[10]], inf_hist_list[[11]], inf_hist_list[[12]],
          ncol = 4, rel_widths = c(1, 0.9, 0.9, 0.9), rel_heights = c(0.15, 1, 1, 1))

# Equilibrium analysis

init_cond_list <- c("most_leaders", "most_followers", "uniform", "a2b1", "a1b2", "a2b2", "a0.75b0.75")

ct_list <- c(3, -3)
sim.vector <- 1:5
criterion_list <- c("sd", "prop")

eq_plot_list <- list()

annotate_df <- data.frame(a = c(0.16, 1.0, 4.0),
                          b = c(4, 0.5, 0.12),
                          text = c("Beta(0.1,1.9)", "Beta(1,1)", "Beta(1.9,0.1)"))

segment_df <- data.frame(a = c(0.103, 1.0),
                         b = c(2.08, 0.95),
                         a_end = c(0.15, 1.0),
                         b_end = c(3.55, 0.55))

for (c_id in 1:length(criterion_list)) {
  criterion <- criterion_list[c_id]

  for (i in 1:length(ct_list)) {
    
    ct <- ct_list[i]
    print(paste("ct =", ct))
    
    all_beta <- data.frame()
  
    for (j in 1:length(init_cond_list)) {
      
      init_cond <- init_cond_list[j]
      all_sim_beta <- data.frame()
      
      for (k in sim.vector) {
        global_beta_path <- file.path(box_path, 'HierarchyWisdom', 'results',
                                      paste0('global_beta_ct', ct, '_',
                                             criterion, '_', init_cond,
                                             '_sim', k, '.csv'))
        global_beta <- read_csv(global_beta_path, show_col_types = FALSE)
        global_beta$sim <- k
        all_sim_beta <- rbind(all_sim_beta, global_beta)
      }
      
      all_sim_beta <- all_sim_beta %>%
        filter(step <= 5000) %>% 
        mutate(log_a = log(a), log_b = log(b),
               init_cond = init_cond)
      
      all_beta <- rbind(all_beta, all_sim_beta)
      
    }
    
    agg_beta <- all_beta %>%
      group_by(init_cond, step) %>%
      summarise(a = mean(a), b = mean(b),
                log_a = mean(log_a), log_b = mean(log_b))
    
    first_steps <- all_beta %>% filter(step == 0) %>% select(-step, -sim, -init_cond)
    first_steps$position <- 'start'
    last_steps <- all_beta %>% filter(step == 5000) %>% select(-step, -sim, -init_cond)
    last_steps$position <- 'end'
    
    agg_first_last <- agg_beta %>% 
      filter(step == 0 | step == 5000) %>% 
      mutate(position = factor(if_else(step == 0, "start", "end"))) %>% 
      mutate(position=fct_relevel(position,c("start","end")))
    
    # lfirst_steps <- first_steps %>% select(log_a, log_b, position)
    # llast_steps <- last_steps %>% select(log_a, log_b, position)
    # 
    # lglobal_first_last <- rbind(rbind(lfirst_steps, llast_steps), agg_first_last)
    
    # visualize distributions of influence based on beta distribution parameters
    eq_plot <- ggplot(all_beta, aes(x=a, y=b, group=interaction(sim, init_cond))) +
      geom_abline(intercept=0, slope=1, linetype="dashed") +
      # geom_hline(yintercept = 0, linetype="dashed") +
      # geom_vline(xintercept = 0, linetype="dashed") +
      geom_path(linewidth=0.2, alpha=0.2) +
      geom_path(data=agg_beta, aes(x=a, y=b, group=init_cond), linewidth=0.75, color='black') +
      geom_point(data=agg_first_last, aes(x=a, y=b, color=position, group=NA), size=2.5, alpha=0.7) +
      geom_text(data=annotate_df, aes(x=a, y=b, label=text, group=NA), size=4.5) +
      geom_segment(data=segment_df, aes(x=a, y=b, xend=a_end, yend=b_end, group=NA)) +
      scale_color_manual(values=c("#00BFC4", "#F8766D")) +
      scale_x_continuous(expression(a), expand=c(0,0), limits = c(1/12, 12), trans = 'log2') +
      scale_y_continuous(expression(b), expand=c(0,0), limits = c(1/12, 12), trans = 'log2') +
      # ggtitle(paste0('(C = ', ct, ', decision rule = ', criterion, ')')) +
      theme_classic() +
      theme(axis.title=element_text(size=16,face="bold"),
            axis.text=element_text(size=14),
            legend.text = element_text(size=14),
            legend.position=c(.15,.45),
            legend.title=element_blank())
    
    plot_index <- i+(length(ct_list)*(c_id-1))
    
    if (plot_index > 1) {
      eq_plot <- eq_plot + theme(legend.position = "none")
    }
    
    print(paste("plot index =", plot_index))
    eq_plot_list[[plot_index]] <- eq_plot
    
  }
  
}

# extract shared legend from the first plot
shared_legend <- get_legend(eq_plot_list[[1]])
eq_plot1 <- eq_plot_list[[1]] + theme(legend.position = "none")

# create labels for each column
column_title_list <- list()

cpos_title <- ggdraw() + 
  draw_label(
    expression(paste("Rapid decision favored (C = ", +3, ")")),
    fontface = 'bold',
    size = 18,
    x = 0.5,
    vjust = 0.5,
    angle = 0
  )

column_title_list[[1]] <- cpos_title

cneg_title <- ggdraw() + 
  draw_label(
    expression(paste("Extended deliberation favored (C = ", -3, ")")),
    fontface = 'bold',
    size = 18,
    x = 0.5,
    vjust = 0.5,
    angle = 0
  )

column_title_list[[2]] <- cneg_title


stend_ct3_grid <- plot_grid(NA, start_col, end_col,
                            NA, inf_hist_list[[1]], inf_hist_list[[2]],
                            NA, inf_hist_list[[5]], inf_hist_list[[6]],
                            NA, inf_hist_list[[9]], inf_hist_list[[10]],
                            ncol = 3, rel_widths = c(0.1, 1, 1), rel_heights = c(0.25, 1, 1, 1.2),
                            label_size = 16,
                            labels = c("a", NA,
                                       NA, NA,
                                       NA, NA))

stend_ctn3_grid <- plot_grid(NA, start_col, end_col,
                             NA, inf_hist_list[[3]], inf_hist_list[[4]],
                             NA, inf_hist_list[[7]], inf_hist_list[[8]],
                             NA, inf_hist_list[[11]], inf_hist_list[[12]],
                             ncol = 3, rel_widths = c(0.24, 1, 1), rel_heights = c(0.25, 1, 1, 1.2),
                             label_size = 16,
                             labels = c("b", NA, NA,
                                        NA, NA, NA,
                                        NA, NA, NA))


plot_grid(
  column_title_list[[1]], column_title_list[[2]],
  stend_ct3_grid, stend_ctn3_grid,
  NA, NA,
  eq_plot_list[[1]], eq_plot_list[[2]],
  label_size = 16,
  ncol = 2, rel_widths = c(1, 1), rel_heights = c(0.1, 0.8, 0.1, 1),
  labels = c(NA, NA,
             NA, NA,
             NA, NA,
             "c", "d")
)


ggsave(file.path(wd_result_path,
                 paste0("diff_organizations_sd.jpg")),
       height = 8, width = 12)







# Supplemental figures

# Equilibrium analysis under all decision rules

init_cond_list <- c("most_leaders", "most_followers", "uniform", "a2b1", "a1b2", "a2b2", "a0.75b0.75")

ct_list <- c(3, -3)
sim.vector <- 1:5
criterion_list <- c("sd", "prop")

eq_plot_list <- list()

annotate_df <- data.frame(a = c(0.16, 1.0, 4.0),
                          b = c(4, 0.5, 0.12),
                          text = c("Beta(0.1,1.9)", "Beta(1,1)", "Beta(1.9,0.1)"))

segment_df <- data.frame(a = c(0.103, 1.0),
                         b = c(2.08, 0.95),
                         a_end = c(0.15, 1.0),
                         b_end = c(3.55, 0.55))

for (c_id in 1:length(criterion_list)) {
  criterion <- criterion_list[c_id]
  
  for (i in 1:length(ct_list)) {
    
    ct <- ct_list[i]
    print(paste("ct =", ct))
    
    all_beta <- data.frame()
    
    for (j in 1:length(init_cond_list)) {
      
      init_cond <- init_cond_list[j]
      all_sim_beta <- data.frame()
      
      for (k in sim.vector) {
        global_beta_path <- file.path(box_path, 'HierarchyWisdom', 'results',
                                      paste0('global_beta_ct', ct, '_',
                                             criterion, '_', init_cond,
                                             '_sim', k, '.csv'))
        global_beta <- read_csv(global_beta_path, show_col_types = FALSE)
        global_beta$sim <- k
        all_sim_beta <- rbind(all_sim_beta, global_beta)
      }
      
      all_sim_beta <- all_sim_beta %>%
        filter(step <= 5000) %>% 
        mutate(log_a = log(a), log_b = log(b),
               init_cond = init_cond)
      
      all_beta <- rbind(all_beta, all_sim_beta)
      
    }
    
    agg_beta <- all_beta %>%
      group_by(init_cond, step) %>%
      summarise(a = mean(a), b = mean(b),
                log_a = mean(log_a), log_b = mean(log_b))
    
    first_steps <- all_beta %>% filter(step == 0) %>% select(-step, -sim, -init_cond)
    first_steps$position <- 'start'
    last_steps <- all_beta %>% filter(step == 5000) %>% select(-step, -sim, -init_cond)
    last_steps$position <- 'end'
    
    agg_first_last <- agg_beta %>% 
      filter(step == 0 | step == 5000) %>% 
      mutate(position = factor(if_else(step == 0, "start", "end"))) %>% 
      mutate(position=fct_relevel(position,c("start","end")))
    
    # lfirst_steps <- first_steps %>% select(log_a, log_b, position)
    # llast_steps <- last_steps %>% select(log_a, log_b, position)
    # 
    # lglobal_first_last <- rbind(rbind(lfirst_steps, llast_steps), agg_first_last)
    
    # visualize distributions of influence based on beta distribution parameters
    eq_plot <- ggplot(all_beta, aes(x=a, y=b, group=interaction(sim, init_cond))) +
      geom_abline(intercept=0, slope=1, linetype="dashed") +
      # geom_hline(yintercept = 0, linetype="dashed") +
      # geom_vline(xintercept = 0, linetype="dashed") +
      geom_path(linewidth=0.2, alpha=0.2) +
      geom_path(data=agg_beta, aes(x=a, y=b, group=init_cond), linewidth=0.75, color='black') +
      geom_point(data=agg_first_last, aes(x=a, y=b, color=position, group=NA), size=2.5, alpha=0.7) +
      geom_text(data=annotate_df, aes(x=a, y=b, label=text, group=NA), size=4.5) +
      geom_segment(data=segment_df, aes(x=a, y=b, xend=a_end, yend=b_end, group=NA)) +
      scale_color_manual(values=c("#00BFC4", "#F8766D")) +
      scale_x_continuous(expression(a), expand=c(0,0), limits = c(1/12, 12), trans = 'log2') +
      scale_y_continuous(expression(b), expand=c(0,0), limits = c(1/12, 12), trans = 'log2') +
      # ggtitle(paste0('(C = ', ct, ', decision rule = ', criterion, ')')) +
      theme_classic() +
      theme(axis.title=element_text(size=16,face="bold"),
            axis.text=element_text(size=14),
            legend.text = element_text(size=14),
            legend.position=c(.15,.45),
            legend.title=element_blank())
    
    plot_index <- i+(length(ct_list)*(c_id-1))
    
    if (plot_index > 1) {
      eq_plot <- eq_plot + theme(legend.position = "none")
    }
    
    print(paste("plot index =", plot_index))
    eq_plot_list[[plot_index]] <- eq_plot
    
  }
  
}

# extract shared legend from the first plot
shared_legend <- get_legend(eq_plot_list[[1]])
eq_plot1 <- eq_plot_list[[1]] + theme(legend.position = "none")

# create labels for each column
column_title_list <- list()

cpos_title <- ggdraw() + 
  draw_label(
    expression(paste("Rapid decision favored (C = ", +3, ")")),
    fontface = 'bold',
    size = 18,
    x = 0.5,
    vjust = 0.5,
    angle = 0
  )

column_title_list[[1]] <- cpos_title

cneg_title <- ggdraw() + 
  draw_label(
    expression(paste("Extended deliberation favored (C = ", -3, ")")),
    fontface = 'bold',
    size = 18,
    x = 0.5,
    vjust = 0.5,
    angle = 0
  )

column_title_list[[2]] <- cneg_title


# create labels for each row (decision rule)
rule_list <- c("Consensus Rule", "Proportional voting")
row_title_list <- list()

for (i in 1:length(criterion_list)) {
  row_title <- ggdraw() +
    draw_label(
      rule_list[i],
      fontface = 'bold',
      size = 14,
      y = 0.5,
      hjust = 0.5,
      angle = 90
    )

  row_title_list[[i]] <- row_title
}

plot_grid(
  NA, column_title_list[[1]], column_title_list[[2]],
  row_title_list[[1]], eq_plot_list[[1]], eq_plot_list[[2]],
  row_title_list[[2]], eq_plot_list[[3]], eq_plot_list[[4]],
  label_size = 18,
  ncol = 3, rel_widths = c(0.1, 1, 1), rel_heights = c(0.1, 1, 1),
  labels = c(NA, NA, NA,
             NA, "a", "b",
             NA, "c", "d")
)

ggsave(file.path(wd_result_path,
                 paste0("diff_organizations.jpg")),
       height = 9, width = 12)






# varying group size

init_cond_list <- c("uniform")

ct_list <- c(3, -3)
gs.vector <- c("50", "500", "800")
gs.sim.map <- list("50" = c(41:45), "500" = c(1:5), "800" = c(61:65))
criterion <- "sd"

eq_plot_list <- list()

annotate_df <- data.frame(a = c(0.16, 1.0, 4.0),
                          b = c(4, 0.5, 0.12),
                          text = c("Beta(0.1,1.9)", "Beta(1,1)", "Beta(1.9,0.1)"))

segment_df <- data.frame(a = c(0.103, 1.0),
                         b = c(2.08, 0.95),
                         a_end = c(0.15, 1.0),
                         b_end = c(3.55, 0.55))


for (i in 1:length(ct_list)) {
  
  ct <- ct_list[i]
  print(paste("ct =", ct))
  
  for (gs.id in 1:length(gs.vector)){
    
    gs <- gs.vector[gs.id]
    sim.vector <- gs.sim.map[[gs]]
    
    all_beta <- data.frame()
  
    for (j in 1:length(init_cond_list)) {
      
      init_cond <- init_cond_list[j]
      all_sim_beta <- data.frame()
      
      for (k in sim.vector) {
        global_beta_path <- file.path(box_path, 'HierarchyWisdom', 'results',
                                      paste0('global_beta_ct', ct, '_',
                                             criterion, '_', init_cond,
                                             '_sim', k, '.csv'))
        global_beta <- read_csv(global_beta_path, show_col_types = FALSE)
        global_beta$sim <- k
        all_sim_beta <- rbind(all_sim_beta, global_beta)
      }
      
      all_sim_beta <- all_sim_beta %>%
        filter(step <= 5000) %>% 
        mutate(log_a = log(a), log_b = log(b),
               init_cond = init_cond)
      
      all_beta <- rbind(all_beta, all_sim_beta)
      
    }
    
    agg_beta <- all_beta %>%
      group_by(init_cond, step) %>%
      summarise(a = mean(a), b = mean(b),
                log_a = mean(log_a), log_b = mean(log_b))
    
    first_steps <- all_beta %>% filter(step == 0) %>% select(-step, -sim, -init_cond)
    first_steps$position <- 'start'
    last_steps <- all_beta %>% filter(step == 5000) %>% select(-step, -sim, -init_cond)
    last_steps$position <- 'end'
    
    agg_first_last <- agg_beta %>% 
      filter(step == 0 | step == 5000) %>% 
      mutate(position = factor(if_else(step == 0, "start", "end"))) %>% 
      mutate(position=fct_relevel(position,c("start","end")))
    
    # visualize distributions of influence based on beta distribution parameters
    eq_plot <- ggplot(all_beta, aes(x=a, y=b, group=interaction(sim, init_cond))) +
      geom_abline(intercept=0, slope=1, linetype="dashed") +
      geom_path(linewidth=0.2, alpha=0.2) +
      geom_path(data=agg_beta, aes(x=a, y=b, group=init_cond), linewidth=0.75, color='black') +
      geom_point(data=agg_first_last, aes(x=a, y=b, color=position, group=NA), size=2.5, alpha=0.7) +
      # geom_text(data=annotate_df, aes(x=a, y=b, label=text, group=NA), size=4.5) +
      # geom_segment(data=segment_df, aes(x=a, y=b, xend=a_end, yend=b_end, group=NA)) +
      scale_color_manual(values=c("#00BFC4", "#F8766D")) +
      scale_x_continuous(expression(a), expand=c(0,0), trans = 'log2') +
      scale_y_continuous(expression(b), expand=c(0,0), trans = 'log2') +
      # ggtitle(paste0('(C = ', ct, ', decision rule = ', criterion, ')')) +
      theme_classic() +
      theme(axis.title=element_text(size=16,face="bold"),
            axis.text=element_text(size=14),
            legend.text = element_text(size=14),
            legend.position=c(.15,.8),
            legend.title=element_blank())
    
    plot_index <- gs.id + (length(gs.vector)*(i-1))
    
    if (plot_index > 1) {
      eq_plot <- eq_plot + theme(legend.position = "none")
    }
    
    print(paste("plot index =", plot_index))
    eq_plot_list[[plot_index]] <- eq_plot
  
  }
}

# extract shared legend from the first plot
shared_legend <- get_legend(eq_plot_list[[1]])
eq_plot1 <- eq_plot_list[[1]] + theme(legend.position = "none")

# create labels for each column
column_title_list <- list()

for (i in 1:length(gs.vector)) {
  column_title <- ggdraw() +
    draw_label(
      paste("group size =", gs.vector[i]),
      fontface = 'bold',
      size = 14,
      x = 0.6,
      vjust = 0.5,
      angle = 0
    )
  
  column_title_list[[i]] <- column_title
}

# create labels for each row (decision rule)
row_title_list <- list()

cpos_title <- ggdraw() + 
  draw_label(
    bquote(atop("Rapid decision favored",
                "(C = " ~ +3 ~ ")")),
    fontface = 'plain',
    size = 14,
    y = 0.6,
    hjust = 0.5,
    angle = 90
  )

row_title_list[[1]] <- cpos_title

cneg_title <- ggdraw() + 
  draw_label(
    bquote(atop("Extended deliberation favored",
                "(C = " ~ -3 ~ ")")),
    fontface = 'plain',
    size = 14,
    y = 0.6,
    hjust = 0.5,
    angle = 90
  )

row_title_list[[2]] <- cneg_title


plot_grid(
  NA, column_title_list[[1]], column_title_list[[2]], column_title_list[[3]],
  row_title_list[[1]], eq_plot_list[[1]], eq_plot_list[[2]], eq_plot_list[[3]],
  row_title_list[[2]], eq_plot_list[[4]], eq_plot_list[[5]], eq_plot_list[[6]],
  label_size = 18,
  ncol = 4, rel_widths = c(0.15, 1, 1, 1), rel_heights = c(0.1, 1, 1),
  labels = c(NA, NA, NA, NA,
             NA, "a", "b", "c",
             NA, "d", "e", "f")
)

ggsave(file.path(wd_result_path,
                 paste0("diff_group_size.jpg")),
       height = 6, width = 12)







# Skewness across a range of C (speed-quality tradeoff)

init_cond_list <- c("uniform") #c("most_leaders", "uniform", "most_followers")

ct_list <- -3:3
sim.vector <- 1:5
criterion <- "sd"

all_beta <- data.frame()

for (i in 1:length(ct_list)) {  
  
  ct <- ct_list[i]
  
  for (j in 1:length(init_cond_list)) {
    
    init_cond <- init_cond_list[j]
    all_sim_beta <- data.frame()
    
    for (k in sim.vector) {
      global_beta_path <- file.path(box_path, 'HierarchyWisdom', 'results',
                                    paste0('global_beta_ct', ct, '_',
                                           criterion, '_', init_cond,
                                           '_sim', k, '.csv'))
      global_beta <- read_csv(global_beta_path, show_col_types = FALSE)
      global_beta$sim <- k
      all_sim_beta <- rbind(all_sim_beta, global_beta)
    }
    
    all_sim_beta <- all_sim_beta %>%
      filter(step <= 5000) %>% 
      mutate(log_a = log(a), log_b = log(b),
             init_cond = init_cond,
             ct = ct)
    
    all_beta <- rbind(all_beta, all_sim_beta)
    
  }
}

sim_skew <- all_beta %>%
  mutate(skew = (2*(b - a)*sqrt(a+b+1)) / ((a+b+2)*sqrt(a*b))) %>% 
  filter(step > 4900, init_cond == "uniform") %>% 
  group_by(ct, init_cond, sim) %>%
  summarise(a = mean(a), b = mean(b),
            log_a = mean(log_a), log_b = mean(log_b),
            skew = mean(skew))

agg_skew <- sim_skew %>% 
  group_by(ct, init_cond) %>% 
  summarise(skew.avg = mean(skew), skew.std = sd(skew), sim_n = n()) %>% 
  mutate(std_err = skew.std/sqrt(sim_n),
         ct.factor = as.factor(ct))

# skew_plot <- ggplot(agg_beta, aes(x = ct, y = skew)) +
#   geom_point() +
#   theme_classic()

skew_plot <- agg_skew %>% 
  ggplot(aes(x=ct.factor, y=skew.avg)) +
  geom_point(alpha=0.7) +
  geom_errorbar(aes(ymin=skew.avg-(2*std_err),
                    ymax=skew.avg+(2*std_err)),
                width=.3) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  # scale_x_discrete("C") +
  scale_y_continuous("Influence Skewness") +
  # scale_color_manual(labels = c("Most Followers", "Most Leaders", "Uniform"),
  #                    values = c("#F8766D", "#7CAE00", "#00BFC4")) +
  # guides(colour = guide_legend(title = "Initial Condition")) +
  theme_classic() +
  theme(axis.title=element_text(size=16,face="bold"),
        axis.title.x = element_blank(),
        axis.text=element_text(size=14),
        legend.title=element_text(size=16),
        legend.text = element_text(size=14),
        legend.position = c(.2, .85),
        legend.box.background = element_rect(colour = "black"))


axis.gg <- ggplot() +
  annotate("segment", y=0.5,yend=0.5,x=-0.05,xend=3.1,arrow=arrow(type = "closed", ends="both"), linewidth=1.5) +
  # annotate("segment", x=c(.5, 1.5, 2.5),xend=c(.5, 1.5, 2.5),y=.49,yend=.51) +
  annotate("text", x=c(.6, 1.57, 2.55), y=0.4, angle=0,
           label = c(expression("Extended deliberation favored"),
                     "C",
                     expression("Rapid decision favored")),
           size = 5.5) +
  scale_x_continuous(limits = c(-0.15, 3.17), expand = c(0,0)) +
  scale_y_continuous(limits = c(0.2, 0.57)) +
  theme_classic() +
  theme(axis.line = element_blank(),
        axis.title = element_blank(),
        axis.ticks = element_blank(),
        axis.text = element_blank())
axis.gg

x_axis_panel <- plot_grid(NA, axis.gg, ncol = 2,
                          rel_widths = c(0.05, 1))

skewness_result_fig <- plot_grid(skew_plot, x_axis_panel, ncol = 1,
                                 rel_heights = c(1, 0.2))


ggsave(file.path(wd_result_path,
                 paste0("skewness_across_c.jpg")),
       skewness_result_fig,
       height = 6, width = 10)






# global skewness sensitivity

ct_list <- c(-3, 3)
sim_digits <- c(0, 30, 50)
sim.vector <- 1:5
criterion <- "sd"

all_beta <- data.frame()

# Consensus rule
for (i in 1:length(sim_digits)) {  
  
  sim_digit <- sim_digits[i]
  
  for (j in 1:length(ct_list)) {
    
    ct <- ct_list[j]
    all_sim_beta <- data.frame()
    
    for (k in sim.vector) {
      global_beta_path <- file.path(box_path, 'HierarchyWisdom', 'results',
                                    paste0('global_beta_ct', ct, '_',
                                           criterion, '_', 'uniform',
                                           '_sim', (k + sim_digit), '.csv'))
      global_beta <- read_csv(global_beta_path, show_col_types = FALSE)
      global_beta$sim <- k
      all_sim_beta <- rbind(all_sim_beta, global_beta)
    }
    
    all_sim_beta <- all_sim_beta %>%
      filter(step <= 5000) %>% 
      mutate(log_a = log(a), log_b = log(b),
             sim_digit = sim_digit, ct = ct)
    
    all_beta <- rbind(all_beta, all_sim_beta)
  }
}

# Proportional voting rule
criterion <- "prop"
for (i in 1:2) {  
  
  sim_digit <- sim_digits[i]
  
  for (j in 1:length(ct_list)) {
    
    ct <- ct_list[j]
    all_sim_beta <- data.frame()
    
    for (k in sim.vector) {
      global_beta_path <- file.path(box_path, 'HierarchyWisdom', 'results',
                                    paste0('global_beta_ct', ct, '_',
                                           criterion, '_', 'uniform',
                                           '_sim', (k + sim_digit), '.csv'))
      global_beta <- read_csv(global_beta_path, show_col_types = FALSE)
      global_beta$sim <- k
      all_sim_beta <- rbind(all_sim_beta, global_beta)
    }
    
    all_sim_beta <- all_sim_beta %>%
      filter(step <= 5000) %>% 
      mutate(log_a = log(a), log_b = log(b),
             sim_digit = sim_digit + 100, ct = ct)
    
    all_beta <- rbind(all_beta, all_sim_beta)
  }
}

sim_skew <- all_beta %>%
  mutate(skew = (2*(b - a)*sqrt(a+b+1)) / ((a+b+2)*sqrt(a*b))) %>% 
  filter(step > 4900) %>% 
  group_by(ct, sim_digit, sim) %>%
  summarise(a = mean(a), b = mean(b),
            log_a = mean(log_a), log_b = mean(log_b),
            skew = mean(skew))

param_set_skew <- sim_skew %>% 
  group_by(ct, sim_digit) %>% 
  summarise(skew.avg = mean(skew), skew.std = sd(skew), sim_n = n()) %>% 
  mutate(std_err = skew.std/sqrt(sim_n), ct.factor = as.factor(ct))

agg_skew <- param_set_skew %>% 
  group_by(ct) %>% 
  summarise(skew.std = sd(skew.avg), skew.avg = mean(skew.avg), config_n = n()) %>% 
  mutate(std_err = skew.std/sqrt(config_n), ct.factor = as.factor(ct))

# skew_plot <- ggplot(agg_beta, aes(x = ct, y = skew)) +
#   geom_point() +
#   theme_classic()

sensitivity_plot <- param_set_skew %>% 
  ggplot(aes(x=ct.factor, y=skew.avg, color=as.factor(sim_digit), group=as.factor(sim_digit))) +
  geom_point(alpha=0.7, position=position_dodge(width=0.3)) +
  # geom_point(data=agg_skew, aes(x=ct.factor, y=skew.avg), size=2, color="black") +
  geom_errorbar(aes(ymin=skew.avg-(2*std_err),
                    ymax=skew.avg+(2*std_err)),
                width=.1, position=position_dodge(width=0.3)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_x_discrete(labels = expression("Extended deliberation favored (C = "*-3*")",
                                       "Rapid decision favored (C = "*+3*")")) +
  scale_y_continuous("Influence Skewness", limits = c(-0.7, 0.7)) +
  scale_color_manual(labels = expression("Consensus Rule (default)", "Consensus Rule ("*x[theta] == 0.1*")",
                                         "Consensus Rule ("*q == 1*")", "Proportional Voting (default)",
                                         "Proportional Voting ("*lambda==0.01*")"),
                     values = hue_pal()(5)) +
  guides(colour = guide_legend(title = "Parameter Configuration")) +
  theme_classic() +
  theme(axis.title=element_text(size=16,face="bold"),
        axis.title.x=element_blank(),
        axis.text=element_text(size=14),
        legend.title=element_text(size=16),
        legend.text = element_text(size=14),
        legend.position = c(.2, .75),
        legend.text.align = 0,
        legend.box.background = element_rect(colour = "black"))

ggsave(file.path(wd_result_path,
                 paste0("param_sensitivity_skew.jpg")),
       sensitivity_plot,
       height = 6, width = 10)


# plot_grid(skew_plot, NA, sensitivity_plot, 
#           ncol = 3, rel_widths = c(1, 0.1, 0.7))
# 
# ggsave(file.path(wd_result_path,
#                  paste0("global_sensitivity_skew.jpg")),
#        height = 6, width = 12)








# Decision time with different leader types
sim <- 1
init_cond_labels <- c("Mostly Followers", "Uniform", "Mostly Leaders")

# generate heatmap for 2 different decision rules
ltype_decis_path <- file.path(box_path, 'HierarchyWisdom', 'results',
                              paste0('opf_sd_leader_type_dtime', sim ,'.csv'))
  
ltype_decis_df <- read_csv(ltype_decis_path) %>% 
  mutate(init_cond = factor(init_cond, levels=c("most_followers", "uniform", "most_leaders")),
         leader_type = factor(leader_type, levels=c("no_leader", "talkative", "persuasive", "influential")),
         std_err = sd_n_event/10)

ltype_decis_df %>% 
  ggplot(aes(x=init_cond, y=avg_n_event, color=leader_type)) +
  geom_point(alpha=0.7, position=position_dodge(width=0.1)) +
  geom_errorbar(aes(ymin=avg_n_event-(2*std_err),
                    ymax=avg_n_event+(2*std_err)),
                width=.3, position=position_dodge(width=0.1)) +
  scale_x_discrete("Distribution of Influence", labels = init_cond_labels) +
  scale_y_continuous("Decision Time", limits = c(0.06,4700), expand=c(0,0)) +
  scale_color_manual(labels = c("no leader", "talkative", "persuasive", "talkative & persuasive"),
                     values = c("#F8766D", "#7CAE00", "#00BFC4", "#C77CFF")) +
  guides(colour = guide_legend(title = "Leader Type")) +
  theme_classic() +
  theme(axis.title=element_text(size=16,face="bold"),
        axis.text=element_text(size=14),
        legend.title=element_text(size=16),
        legend.text = element_text(size=14),
        legend.position = c(.2, .85),
        legend.box.background = element_rect(colour = "black"))

ggsave(file.path(wd_result_path,
                 paste0("leader_type_time_plot.jpg")),
       height = 6, width = 10)






library(raster)

# Group-level payoff heatmaps (derived from decision time)
lim_speakers <- 1
criterion_list <- c('sd', 'prop')

heatmap_list <- list()

beta_sample_locs <- data.frame(a = c(0.1, 1, 1.9),
                               b = c(1.9, 1, 0.1))

annotate_df <- data.frame(a = c(0.5, 1.25, 1.6),
                          b = c(1.9, 1.1, 0.25),
                          text = c("Beta(0.1,1.9)", "Beta(1,1)", "Beta(1.9,0.1)"))

# generate heatmap for 2 different decision rules
for (i in 1:2) {
  criterion <- criterion_list[i]
  
  beta_decis_path <- file.path(box_path, 'HierarchyWisdom', 'results',
                               paste0('opfsp', lim_speakers, '_', criterion,
                                      '_beta_decision_time.csv'))
  
  beta_decis_df <- read_csv(beta_decis_path) %>% 
    mutate(cpos_payoff = clamp(5000 - (3*avg_n_event), lower=0),
           cneg_payoff = 3*avg_n_event)
  
  # 5000 - 3*median(beta_decis_df$avg_n_event)
  # 3*median(beta_decis_df$avg_n_event)
  
  # Alternative colors: low: #56B4E9, high: #E69F00
  max_payoff <- 13800 # max payoff of consensus rule with c = -3
  
  colors <- c("blue", "white", "red")
  color_breaks <- c(0, 2500, max_payoff)
  
  cpos_heatmap <- beta_decis_df %>% ggplot(aes(x=a, y=b)) + 
    geom_tile(aes(fill=cpos_payoff)) +
    # geom_contour(aes(z=value),
    #              color="black",
    #              breaks = c(0.4, 1, 1.8, 2.4)) +
    geom_point(data = beta_sample_locs, color = "#212f3c", size = 4) +
    geom_text(data = annotate_df, aes(x=a, y=b, label=text),
              size=5 , fontface="bold") +
    scale_fill_gradientn(limits = c(0, max_payoff), 
                         colors = colors,
                         values = scales::rescale(color_breaks)) +
    geom_hline(yintercept = 1) +
    geom_vline(xintercept = 1) +
    scale_x_continuous(expression(a), limits = c(0.06,2.03), expand=c(0,0)) +
    scale_y_continuous(expression(b), limits = c(0.06,2.03), expand=c(0,0)) +
    guides(fill = guide_colourbar(title = "Group payoff")) +
    theme_classic() +
    theme(panel.background = element_blank(),
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          axis.title=element_text(size=18,face="bold"),
          axis.text=element_text(size=14),
          legend.key.width  = unit(1, "lines"),
          legend.key.height = unit(2, "lines"),
          axis.line = element_line(colour = "black"),
          panel.border = element_rect(colour = "black",
                                      fill=NA, linewidth=3))
  
  cneg_heatmap <- beta_decis_df %>% ggplot(aes(x=a, y=b)) + 
    geom_tile(aes(fill=cneg_payoff)) +
    # geom_contour(aes(z=value),
    #              color="black",
    #              breaks = c(0.4, 1, 1.8, 2.4)) +
    geom_point(data = beta_sample_locs, color = "#212f3c", size = 4) +
    geom_text(data = annotate_df, aes(x=a, y=b, label=text),
              size=5 , fontface="bold") +
    scale_fill_gradientn(limits = c(0, max_payoff), 
                         colors = colors,
                         values = scales::rescale(color_breaks)) +
    geom_hline(yintercept = 1) +
    geom_vline(xintercept = 1) +
    scale_x_continuous(expression(a), limits = c(0.06,2.03), expand=c(0,0)) +
    scale_y_continuous(expression(b), limits = c(0.06,2.03), expand=c(0,0)) +
    guides(fill = guide_colourbar(title = "Group payoff")) +
    theme_classic() +
    theme(panel.background = element_blank(),
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          axis.title=element_text(size=18,face="bold"),
          axis.text=element_text(size=14),
          legend.key.width  = unit(1, "lines"),
          legend.key.height = unit(2, "lines"),
          axis.line = element_line(colour = "black"),
          panel.border = element_rect(colour = "black",
                                      fill=NA, linewidth=3))
  
  plot_index <- 2*(i-1)
  heatmap_list[[plot_index+1]] <- cpos_heatmap
  heatmap_list[[plot_index+2]] <- cneg_heatmap
}

consensus_title <- ggdraw() + 
  draw_label(
    "Consensus Rule",
    fontface = 'bold',
    size = 16,
    x = 0.6,
    vjust = 0.5,
    angle = 0
  )

prop_title <- ggdraw() + 
  draw_label(
    "Proportional Voting",
    fontface = 'bold',
    size = 16,
    x = 0.6,
    vjust = 0.5,
    angle = 0
  )


# create labels for each row (decision rule)
row_title_list <- list()

cpos_title <- ggdraw() + 
  draw_label(
    bquote(atop("Rapid decision favored",
                "(C = " ~ +3 ~ ")")),
    fontface = 'plain',
    size = 14,
    y = 0.6,
    hjust = 0.5,
    angle = 90
  )

row_title_list[[1]] <- cpos_title

cneg_title <- ggdraw() + 
  draw_label(
    bquote(atop("Extended deliberation favored",
                "(C = " ~ -3 ~ ")")),
    fontface = 'plain',
    size = 14,
    y = 0.6,
    hjust = 0.5,
    angle = 90
  )

row_title_list[[2]] <- cneg_title

tmp <- get_legend(heatmap_list[[1]])

heatmap_list[[1]] <- heatmap_list[[1]] + theme(legend.position = "none")
heatmap_list[[2]] <- heatmap_list[[2]] + theme(legend.position = "none")
heatmap_list[[3]] <- heatmap_list[[3]] + theme(legend.position = "none")
heatmap_list[[4]] <- heatmap_list[[4]] + theme(legend.position = "none")


heatmaps <- plot_grid(NA, consensus_title, prop_title,
                      row_title_list[[1]], heatmap_list[[1]], heatmap_list[[3]],
                      row_title_list[[2]], heatmap_list[[2]], heatmap_list[[4]],
                      ncol = 3, labels = c(NA, NA, NA,
                                           NA, "a", "b",
                                           NA, "c", "d"), label_size = 20,
                      rel_widths = c(0.15, 1, 1), rel_heights = c(0.1, 1, 1))

plot_grid(heatmaps, NA, tmp,
          ncol = 3, rel_widths = c(1, 0.03, 0.125))

ggsave(file.path(wd_result_path,
                 paste0("group_payoff_plots.jpg")),
       height = 8, width = 10)




# variance of beta distribution

beta_sample_locs <- data.frame(a = c(0.1, 1, 1.9),
                               b = c(1.9, 1, 0.1))

annotate_df <- data.frame(a = c(0.27, 1.1, 1.8),
                          b = c(1.9, 1.05, 0.15),
                          text = c("Beta(0.1,1.9)", "Beta(1,1)", "Beta(1.9,0.1)"))

criterion <- "sd"
lim_speakers <- 1

beta_decis_path <- file.path(box_path, 'HierarchyWisdom', 'results',
                             paste0('opfsp', lim_speakers, '_', criterion,
                                    '_beta_decision_time.csv'))

# get a and b parameter combinations from simulation results
beta_ab_df <- read_csv(beta_decis_path) %>% 
  as.data.frame() %>% 
  dplyr::select(a, b)

beta_ab_df <- beta_ab_df %>% 
  mutate(beta.var = (a*b) / ((a+b)^2 * (a+b+1)))

max_var <- max(beta_ab_df$beta.var)
mid_var <- median(beta_ab_df$beta.var)

colors <- c("blue", "white", "red")
color_breaks <- c(0, mid_var, max_var)

var_heatmap <- beta_ab_df %>% ggplot(aes(x=a, y=b)) + 
  geom_tile(aes(fill=beta.var)) +
  # geom_contour(aes(z=value),
  #              color="black",
  #              breaks = c(0.4, 1, 1.8, 2.4)) +
  geom_point(data = beta_sample_locs, color = "#212f3c", size = 4) +
  geom_text(data = annotate_df, aes(x=a, y=b, label=text),
            size=5 , fontface="bold") +
  scale_fill_gradientn(limits = c(0, max_var), 
                       colors = colors,
                       values = scales::rescale(color_breaks)) +
  geom_hline(yintercept = 1) +
  geom_vline(xintercept = 1) +
  scale_x_continuous(expression(a), limits = c(0.06,2.03), expand=c(0,0)) +
  scale_y_continuous(expression(b), limits = c(0.06,2.03), expand=c(0,0)) +
  guides(fill = guide_colourbar(title = "Variance")) +
  theme_classic() +
  theme(panel.background = element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.title=element_text(size=18,face="bold"),
        axis.text=element_text(size=14),
        legend.key.width  = unit(1, "lines"),
        legend.key.height = unit(2, "lines"),
        axis.line = element_line(colour = "black"),
        panel.border = element_rect(colour = "black",
                                    fill=NA, linewidth=3))

ggsave(file.path(wd_result_path,
                 paste0("beta_variance_plots.jpg")),
       var_heatmap,
       height = 8, width = 10)

