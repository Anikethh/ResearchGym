## Research Goal
The success of most existing cross-modal retrieval methods heavily relies on the assumption that given queries follow the same distribution as the source domain. However, this assumption is easily violated in real-world scenarios due to the complexity and diversity of queries, leading to the query shift problem. Query shift refers to an online query stream originating from a domain that follows a different distribution than the source, causing significant performance degradation. In real-world applications like search engines, users may have diverse cultural backgrounds or personal preferences, resulting in online queries from scarce or highly personalized domains. These out-of-domain queries violate the identical distribution assumption that pre-trained models rely on. Consequently, existing cross-modal retrieval models fail to handle this query shift and suffer significant performance drops, necessitating an online adaptation method to address this problem.

## Experimental Settings
*   **Source Models**: CLIP (ViT-B/16) and BLIP (ViT-B/16, ViT-L/16).
*   **Datasets & Settings**:
    *   **Query Shift (QS)**: Queries have a different distribution from the gallery. Benchmarks created are COCO-C and Flickr-C, built from COCO and Flickr by adding 16 image corruption types (Noise, Blur, Weather, Digital) and 15 text corruption types (character, word, sentence-level) at various severity levels.
    *   **Query-Gallery Shift (QGS)**: Both query and gallery samples come from distributions different from the source. Datasets include Fashion-Gen (e-commerce), CUHK-PEDES and ICFG-PEDES (person Re-ID)[Primary sub-tasks], and COCO, Flickr [Primary sub-task], Nocaps (natural image).
*   **Testing Protocols**: Image-to-text retrieval (I2TR) and text-to-image retrieval (T2IR).
*   **Evaluation Metric**: Recall@1 (R@1).

## Evaluation Metrics
- T2IR@1(Text-to-image retrieval)
- I2TR@1 (Image-to-text retrieval)

## Hints
- You are expected to setup the datasets and weights directories and populate using the provided links in the README.md file.
- You can use the gdown library to directly download google drive contents, and git clone for studying relevant repositories.

## Baseline Results (to beat)

Table 1: Comparisons with state-of-the-art methods on COCO-C benchmark under QUERY SHIFT ON THE IMAGE MODALITY with maximum severity level regarding the Recall@1 metric.

| Query Shift | Gauss. | Shot | Impul. | Speckle | Defoc. | Glass | Motion | Zoom | Snow | Frost | Fog | Brit. | Contr. | Elastic | Pixel | JPEG | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **BLIP ViT-B/16** | 43.4 | 46.3 | 43.2 | 57.3 | 43.3 | 68.0 | 39.7 | 8.4 | 32.3 | 52.2 | 57.0 | 66.8 | 36.0 | 41.3 | 20.6 | 63.7 | 45.0 |
| • Tent | 41.6 | 40.5 | 37.9 | 54.0 | 44.7 | 65.1 | 39.6 | 8.3 | 31.9 | 48.7 | 56.3 | 66.5 | 31.8 | 40.3 | 19.2 | 62.3 | 43.0 |
| • EATA | 41.4 | 50.3 | 35.7 | 63.1 | 49.8 | 72.2 | 46.2 | 6.9 | 45.6 | 56.7 | 62.5 | 71.4 | 43.6 | 51.3 | 25.6 | 67.0 | 49.3 |
| • SAR | 42.3 | 51.5 | 37.5 | 61.8 | 40.3 | 71.5 | 32.8 | 6.2 | 38.0 | 56.2 | 59.1 | 70.6 | 31.1 | 53.5 | 17.5 | 66.4 | 46.0 |
| • READ | 45.8 | 48.4 | 37.2 | 59.9 | 44.5 | 71.8 | 46.6 | 11.5 | 39.9 | 49.9 | 58.4 | 70.3 | 35.8 | 45.0 | 18.8 | 66.2 | 46.9 |
| • DeYO | 47.9 | 53.5 | 46.8 | 63.4 | 42.9 | 72.1 | 36.7 | 3.2 | 37.5 | 59.7 | 66.4 | 71.2 | 40.3 | 49.0 | 13.1 | 67.6 | 48.2 |
| Your Method | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| **BLIP ViT-L/16** | 50.3 | 51.8 | 51.1 | 61.6 | 53.7 | 72.1 | 49.4 | 14.5 | 44.0 | 57.5 | 61.8 | 70.5 | 37.3 | 50.6 | 32.0 | 70.5 | 51.8 |
| • Tent | 46.3 | 49.3 | 46.7 | 58.4 | 52.2 | 71.8 | 47.5 | 12.3 | 41.9 | 56.2 | 60.9 | 69.7 | 35.7 | 48.3 | 29.4 | 69.6 | 49.8 |
| • EATA | 46.2 | 53.5 | 49.5 | 63.8 | 56.5 | 73.8 | 52.6 | 18.4 | 50.6 | 59.1 | 64.5 | 72.1 | 40.7 | 55.4 | 43.5 | 70.7 | 54.4 |
| • SAR | 45.9 | 50.2 | 47.3 | 63.1 | 51.1 | 73.8 | 47.2 | 11.6 | 40.8 | 58.9 | 60.7 | 71.6 | 33.6 | 54.0 | 34.4 | 70.5 | 50.9 |
| • READ | 38.1 | 48.0 | 43.3 | 63.5 | 43.6 | 73.4 | 43.6 | 22.0 | 44.5 | 56.5 | 62.2 | 71.9 | 32.9 | 49.6 | 27.5 | 70.6 | 49.5 |
| • DeYO | 39.9 | 50.2 | 43.5 | 63.8 | 50.4 | 74.0 | 52.4 | 5.4 | 49.5 | 59.3 | 62.8 | 71.8 | 34.0 | 54.7 | 34.4 | 69.7 | 51.0 |
| Your Method | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |

Table 2: Comparisons with state-of-the-art methods on COCO-C benchmark under QUERY SHIFT ON THE TEXT MODALITY with maximum severity level regarding the Recall@1 metric.

| Query Shift | OCR | CI | CR | CS | CD | SR | RI | RS | RD | IP | Formal | Casual | Passive | Active | Backtrans | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **BLIP ViT-B/16** | 31.4 | 11.3 | 9.4 | 18.9 | 11.4 | 43.6 | 51.5 | 50.3 | 50.6 | 56.8 | 56.6 | 56.2 | 54.9 | 56.8 | 54.2 | 40.9 |
| • Tent | 31.4 | 11.0 | 9.5 | 17.7 | 11.3 | 43.2 | 51.3 | 50.3 | 50.6 | 56.6 | 56.2 | 56.0 | 54.9 | 56.9 | 53.9 | 40.7 |
| • EATA | 33.1 | 11.9 | 10.5 | 18.4 | 12.0 | 44.9 | 53.0 | 51.6 | 50.3 | 56.2 | 56.8 | 56.8 | 56.0 | 56.8 | 54.3 | 41.5 |
| • SAR | 31.8 | 11.6 | 9.9 | 18.5 | 11.7 | 43.6 | 51.5 | 50.3 | 50.6 | 56.8 | 56.5 | 56.2 | 54.9 | 56.8 | 54.2 | 41.0 |
| • READ | 32.3 | 11.4 | 9.6 | 18.2 | 11.2 | 44.3 | 52.9 | 51.7 | 51.1 | 57.6 | 57.1 | 56.7 | 55.9 | 57.1 | 54.7 | 41.4 |
| • DeYO | 31.4 | 11.3 | 9.4 | 17.9 | 11.4 | 43.6 | 51.5 | 50.3 | 50.6 | 56.8 | 56.5 | 56.2 | 54.9 | 56.7 | 54.2 | 40.9 |
| Your Method | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| **BLIP ViT-L/16** | 34.5 | 12.3 | 11.1 | 19.7 | 12.9 | 46.0 | 54.4 | 54.0 | 53.5 | 59.4 | 59.1 | 58.8 | 57.8 | 59.4 | 56.7 | 43.3 |
| • Tent | 34.0 | 12.3 | 11.0 | 19.6 | 12.9 | 46.5 | 54.2 | 53.8 | 53.4 | 59.4 | 59.1 | 58.8 | 57.6 | 58.9 | 56.5 | 43.2 |
| • EATA | 35.6 | 13.3 | 11.3 | 20.3 | 13.2 | 47.2 | 55.4 | 54.2 | 53.8 | 59.2 | 59.1 | 59.4 | 57.9 | 59.4 | 56.8 | 43.7 |
| • SAR | 34.5 | 13.1 | 11.2 | 20.3 | 13.1 | 46.7 | 54.4 | 54.0 | 53.5 | 59.5 | 59.1 | 58.8 | 57.8 | 59.4 | 56.7 | 43.5 |
| • READ | 35.3 | 12.2 | 10.9 | 19.1 | 12.7 | 47.3 | 55.1 | 55.0 | 53.3 | 59.7 | 59.3 | 59.1 | 58.1 | 59.6 | 56.7 | 43.6 |
| • DeYO | 34.5 | 12.3 | 11.1 | 19.7 | 12.9 | 46.7 | 54.4 | 54.0 | 53.5 | 59.5 | 59.1 | 58.8 | 57.8 | 59.4 | 56.7 | 43.4 |
| Your Method | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |

Table 3: Comparisons with state-of-the-art methods on benchmarks under QUERY-GALLERY SHIFTS regarding the Recall@1 metric. In the table, “ID”, “ND” and “OD” refer to “In-Domain”, “Near-Domain” and “Out-Domain”, respectively. Besides, “I2TR@1” / “T2IR@1” represent Recall@1 for image-to-text retrieval / text-to-image retrieval.

| Query Shift | Base2Flickr I2TR@1 | Base2Flickr T2IR@1 | Base2COCO I2TR@1 | Base2COCO T2IR@1 | Base2Fashion I2TR@1 | Base2Fashion T2IR@1 | Base2Nocaps(ID) I2TR@1 | Base2Nocaps(ID) T2IR@1 | Base2Nocaps(ND) I2TR@1 | Base2Nocaps(ND) T2IR@1 | Base2Nocaps(OD) I2TR@1 | Base2Nocaps(OD) T2IR@1 | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **CLIP ViT-B/16** | 80.2 | 61.5 | 52.5 | 33.0 | 8.5 | 13.2 | 84.9 | 61.4 | 75.4 | 49.2 | 73.8 | 55.8 | 54.1 |
| • Tent | 81.4 | 64.0 | 48.8 | 27.6 | 5.6 | 10.7 | 85.1 | 61.7 | 74.6 | 48.6 | 71.8 | 56.1 | 53.0 |
| • EATA | 80.4 | 63.4 | 52.1 | 34.8 | 8.1 | 12.0 | 84.7 | 62.0 | 75.1 | 52.3 | 74.1 | 56.9 | 54.7 |
| • SAR | 80.3 | 62.2 | 51.8 | 33.9 | 8.0 | 13.3 | 84.7 | 61.3 | 75.4 | 51.3 | 73.7 | 56.1 | 54.3 |
| • READ | 80.6 | 64.4 | 46.0 | 35.7 | 5.8 | 11.2 | 85.1 | 63.0 | 75.0 | 52.1 | 73.5 | 57.0 | 54.1 |
| • DeYO | 80.1 | 64.0 | 51.5 | 33.4 | 6.9 | 10.9 | 84.4 | 62.2 | 75.1 | 52.0 | 73.2 | 57.3 | 54.3 |
| Your Method | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| **BLIP ViT-B/16** | 70.0 | 68.3 | 59.3 | 45.4 | 19.9 | 26.1 | 88.2 | 74.9 | 79.3 | 63.6 | 81.9 | 67.8 | 62.1 |
| • Tent | 81.9 | 68.5 | 61.7 | 41.7 | 14.1 | 26.1 | 88.5 | 75.4 | 82.6 | 64.1 | 82.7 | 68.9 | 63.0 |
| • EATA | 82.3 | 69.4 | 64.2 | 47.9 | 12.8 | 25.2 | 87.8 | 75.1 | 82.8 | 63.9 | 81.5 | 67.9 | 63.4 |
| • SAR | 81.7 | 68.3 | 63.5 | 46.6 | 17.9 | 26.1 | 88.2 | 75.6 | 81.0 | 65.4 | 81.2 | 69.3 | 63.7 |
| • READ | 80.0 | 69.9 | 62.1 | 46.4 | 5.6 | 24.1 | 87.3 | 75.1 | 80.6 | 63.9 | 80.7 | 67.9 | 62.0 |
| • DeYO | 83.5 | 69.9 | 65.0 | 47.3 | 12.2 | 24.1 | 89.2 | 75.6 | 83.7 | 65.7 | 84.3 | 69.4 | 64.2 |
| Your Method | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |

Table 4: Comparisons with state-of-the-art methods on ReID benchmarks under QUERY-GALLERY SHIFTS regarding the Recall@1 metric.

| Query Shift | CUHK2ICFG T2IR@1 | ICFG2CUHK T2IR@1 | Avg. |
|---|---|---|---|
| **CLIP ViT-B/16** | 33.3 | 41.0 | 37.2 |
| • Tent | 33.5 | 41.9 | 37.7 |
| • EATA | 33.3 | 42.2 | 37.8 |
| • SAR | 33.3 | 42.2 | 37.8 |
| • READ | 33.0 | 42.3 | 37.7 |
| • DeYO | 33.3 | 42.2 | 37.8 |
| Your Method | -- | -- | -- |