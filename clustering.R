# 필요한 라이브러리 로드
library(data.table)
library(ConsensusClusterPlus)
library(survival)
library(survminer)
library(Rtsne)
library(ggplot2)
# 작업 경로 설정
setwd("C:\\Users\\nsw\\Documents\\_dataset\\BRCA\\")
# 파일 경로 설정
pdf_op_path <- "kmeans_op_folder"
ip_file_path <- "data/AE_latent_TCGA.csv"
input_file_name <- tools::file_path_sans_ext(basename(ip_file_path))  # 입력 파일 이름 추출 (확장자 제거)
pac_op_path <- paste0("Consensus_PAC_values_", input_file_name, ".csv")  # PAC 값 파일 이름에 데이터 정보 포함
# 클러스터 결과 파일 경로 설정
op_files <- file.path(getwd(), paste0("Class_Consensus_Clustering_K_", 10:2, "_", input_file_name, ".csv"))
# 데이터 읽기
ae_data <- fread(ip_file_path, header = TRUE, data.table = FALSE, stringsAsFactors = FALSE)
ae_data_transposed <- t(ae_data[,-1])  # 첫 번째 열 제외하고 전치
colnames(ae_data_transposed) <- ae_data$V1  # 전치된 데이터의 열 이름 설정
rownames(ae_data_transposed) <- colnames(ae_data)[-1]  # 샘플 이름 설정
# 데이터프레임으로 변환
ae_data_final <- as.data.frame(ae_data_transposed)
ae_data_final$V1 <- 0:(nrow(ae_data_final) - 1)
ae_data_final <- ae_data_final[, c(ncol(ae_data_final), 0:(ncol(ae_data_final) - 1))]
# V1 열 제거 (필요 시)
if ("V1" %in% colnames(ae_data_final)) {
  ae_data_final <- ae_data_final[, -1]  # 첫 번째 열(V1) 제거
}
# Consensus Clustering
res.ConClust <- ConsensusClusterPlus(
  data.matrix(ae_data_final), # 클러스터링 대상 데이터
  maxK = 10,                              # 최대 클러스터 수
  reps = 1000,                            # 반복 횟수
  pItem = 0.8,                            # 샘플 비율
  pFeature = 1,                           # 특징 비율
  plot = "pdf",                           # 결과를 PDF로 저장
  title = pdf_op_path,                    # PDF 저장 경로
  distance = "euclidean",                 # 거리 계산 방법
  clusterAlg = "km",                      # 클러스터링 알고리즘
  seed = 2111133                          # 랜덤 시드
)
# 클러스터링 결과 저장
for (k in 10:2) {
  clusters <- as.data.frame(res.ConClust[[k]][["consensusClass"]])
  colnames(clusters) <- "Clusters"
  fwrite(clusters, op_files[11 - k], sep = ",", row.names = TRUE, col.names = TRUE)
}
# PAC 값 계산
maxK <- 10
Kvec <- 2:maxK
x1 <- 0.1
x2 <- 0.9
PAC <- rep(NA, length(Kvec))
names(PAC) <- paste0("K=", Kvec)
for (i in Kvec) {
  M <- res.ConClust[[i]]$consensusMatrix
  Fn <- ecdf(M[lower.tri(M)])
  PAC[i - 1] <- Fn(x2) - Fn(x1)
}
# 최적 K 선택
optK <- Kvec[which.min(PAC)]
cat(sprintf("Optimal K = %d\n", optK))
fwrite(as.data.frame(PAC), pac_op_path, row.names = TRUE, col.names = TRUE)
# 최적 클러스터링 결과 추출
optimal_clusters <- as.data.frame(res.ConClust[[optK]][["consensusClass"]])
colnames(optimal_clusters) <- "Cluster"
optimal_clusters$Sample <- rownames(optimal_clusters)
# t-SNE 적용
compressed_data_transposed <- t(ae_data_final)  # 데이터 전치
rownames(compressed_data_transposed) <- colnames(ae_data_final)  # 샘플 이름 복구
set.seed(42)
tsne_results <- Rtsne(data.matrix(compressed_data_transposed), dims = 2, perplexity = 30, verbose = TRUE, max_iter = 500)
rownames(tsne_results$Y) <- rownames(compressed_data_transposed)
# t-SNE 결과와 클러스터 레이블 매칭
rownames(optimal_clusters) <- optimal_clusters$Sample
common_samples <- intersect(rownames(tsne_results$Y), rownames(optimal_clusters))
# 데이터프레임 생성
tsne_data <- data.frame(
  Dim1 = tsne_results$Y[common_samples, 1],
  Dim2 = tsne_results$Y[common_samples, 2],
  Cluster = as.factor(optimal_clusters[common_samples, "Cluster"])
)
# t-SNE 시각화
ggplot(tsne_data, aes(x = Dim1, y = Dim2, color = Cluster)) +
  geom_point(size = 2, alpha = 0.7) +
  labs(title = paste("t-SNE Visualization for Optimal K =", optK),
       x = "t-SNE Dimension 1", y = "t-SNE Dimension 2") +
  theme_minimal() +
  theme(legend.title = element_text(size = 12),
        legend.text = element_text(size = 10),
        plot.title = element_text(hjust = 0.5, size = 14))