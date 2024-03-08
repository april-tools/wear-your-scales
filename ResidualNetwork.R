#!/usr/bin/env Rscript

library(argparse)

x <- c("comprehenr", "mgm", "qgraph", "bootnet", "OpenMx", "bootnet", "argparse")
new.packages <- x[!(x %in% installed.packages()[, "Package"])]
if (length(new.packages)) {
  install.packages(new.packages, repos = "http://cran.us.r-project.org")
}
lapply(x, require, character.only = TRUE)

residual_network <- function(args) {
  ################## --------------------------------------- ##################
  ################## --------------- Network --------------- ##################
  ################## --------------------------------------- ##################
  
  data <- read.csv(file = args$file)
  
  var_types <- rep(c("g"), each = 28)
  var_levels <- rep(1, 28)
  var_groups <- list(
    "YMRS" = c(1:11),
    "HDRS" = c(12:28)
  )
  var_labels <- c(
    c(to_vec(for (i in c(1:11)) paste("Y", as.character(i), sep = ""))),
    c(to_vec(for (i in c(1:17)) paste("H", as.character(i), sep = "")))
  )
  
  ymrs_item_names <- c(
    "Elevated mood", "Increased motor activity-energy", "Sexual interest", "Sleep",
    "Irritability", "Speech (rate and amount)", "Language-thought disorder",
    "Content", "Disruptive-aggressive behavior", "Appearance", "Insight_y"
  )
  hdrs_item_names <- c(
    "Depressed mood", "Feelings of guilt", "Suicide",
    "Early insomnia", "Middle insomnia",
    "Late insomnia", "Work and activities",
    "Retardation", "Agitation", "Anxiety psychic", "Anxiety somatic",
    "Somatic symptoms gastrointestinal", "General somatic symptoms",
    "Genital symptoms", "Hypochondriasis", "Loss of weight", "Insight_h"
  )
  var_names <- c(ymrs_item_names, hdrs_item_names)
  group_cols <- c("#DC2F02", "#0077B6") #c("#D00000", "#00B4D8") 
  
  data_n <- list(
    "data" = NULL,
    "type" = NULL,
    "level" = NULL,
    "names" = NULL,
    "labels" = NULL,
    "grouplabels" = NULL
  )
  
  # Fill in
  data_n$data <- as.matrix(data)
  data_n$type <- var_types
  data_n$level <- var_levels
  data_n$labels <- var_labels
  data_n$grouplabels <- var_groups
  
  # Fit model
  fit <- mgm(
    data = data_n$data,
    type = data_n$type,
    level = data_n$level,
    lambdaSel = "EBIC",
    lambdaGam = 0.25,
    scale = TRUE
  )
  
  # Compute Predictability
  Pred_cov <- predict(fit, data_n$data)
  Pred_cov$errors
  # predictability estimates
  pie <- as.numeric(as.character(Pred_cov$errors[, 3]))
  mean(pie)
  
  # Plot Network
  
  graph_plot <- qgraph(fit$pairwise$wadj,
                       vsize = 5,
                       layout = "spring",
                       color = group_cols,
                       border.width = 1,
                       border.color = "black",
                       groups = var_groups,
                       nodeNames = var_names,
                       labels = var_labels,
                       legend = FALSE,
                       lcolor = c(rep("white", 28)),
                       cut = 0,
                       pie = pie
  )
  
  
  pdf(
    file = file.path(dirname(args$file), "residual_network.pdf"),
    width = 10, height = 10, pointsize = 18
  )
  plot(graph_plot)
  dev.off()
  
  wadj <- as.data.frame(fit$pairwise$wadj)
  rownames(wadj) <- var_names
  colnames(wadj) <- var_labels
  write.csv(wadj, file.path(dirname(args$file), "wadj.csv"), row.names = TRUE)
  
  ############ --------------------------------------------------- ############
  ############ --------- Re-estimate network via bootnet --------- ############
  ############ --------------------------------------------------- ############
  
  colnames(data_n$data) <- var_labels
  
  # Then re-estimate all networks via bootnet
  n <- estimateNetwork(
    data = data_n$data,
    type = data_n$type,
    level = data_n$level,
    default = "mgm",
    criterion = "EBIC",
    tuning = 0.25
  )
  
  
  ### check if bootnet results match mgm estimation results, using absolute values
  ### for estimateNetwork because this is the way pairwise$wadj is encoded in mgm
  
  # 1, good to go
  stopifnot(cor(vech(fit$pairwise$wadj), vech(abs(n$graph))) == 1)
  
  # Set bootstrap number
  nB <- 500
  
  # Bootstrap:
  
  bs <- bootnet(n, nBoots = nB, nCores = 8)
  
  pdf(file = file.path(dirname(args$file), "bs.pdf"), width = 6, height = 6)
  plot(bs, order = "sample", plot = "area", labels = FALSE)
  dev.off()
  
  pdf(file.path(dirname(args$file), "bs_diff.pdf"), width = 4, height = 4)
  plot(bs,
       order = "sample", plot = "difference", onlyNonZero = TRUE,
       labels = FALSE
  )
  dev.off()
}

# Parser
parser <- ArgumentParser(description = "MGM Residuals")
parser$add_argument("-f", "--file",
                    type = "character", dest = "file",
                    help = "Provide path to residuals file",
                    required = TRUE
)
args <- parser$parse_args()
residual_network(args = args)

# use:
# Rscript ResidualNetwork.R -f runs/<best_model>/residuals.csv