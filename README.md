<div align="center">    
 
# 🌎 GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization

[![Paper](http://img.shields.io/badge/paper-arxiv.2309.16020-B31B1B.svg)](https://arxiv.org/abs/2309.16020v2)
[![Conference](https://img.shields.io/badge/NeurIPS-2023-blue)]()
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/geoclip-clip-inspired-alignment-between/photo-geolocation-estimation-on-im2gps3k)](https://paperswithcode.com/sota/photo-geolocation-estimation-on-im2gps3k?p=geoclip-clip-inspired-alignment-between)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/geoclip-clip-inspired-alignment-between/gps-embeddings-on-geo-tagged-nus-wide-gps)](https://paperswithcode.com/sota/gps-embeddings-on-geo-tagged-nus-wide-gps?p=geoclip-clip-inspired-alignment-between)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/geoclip-clip-inspired-alignment-between/photo-geolocation-estimation-on-gws15k)](https://paperswithcode.com/sota/photo-geolocation-estimation-on-gws15k?p=geoclip-clip-inspired-alignment-between)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/geoclip-clip-inspired-alignment-between/gps-embeddings-on-geo-tagged-nus-wide-gps-1)](https://paperswithcode.com/sota/gps-embeddings-on-geo-tagged-nus-wide-gps-1?p=geoclip-clip-inspired-alignment-between)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/geoclip-clip-inspired-alignment-between/photo-geolocation-estimation-on-yfcc26k)](https://paperswithcode.com/sota/photo-geolocation-estimation-on-yfcc26k?p=geoclip-clip-inspired-alignment-between)

![ALT TEXT](/figures/GeoCLIP.png)

</div>
 
## Description
GeoCLIP addresses the challenges of worldwide image geo-localization by introducing a novel CLIP-inspired approach that aligns images with geographical locations, achieving state-of-the-art results on geo-localization and GPS to vector representation on benchmark datasets (Im2GPS3k, YFCC26k, GWS15k, and the Geo-Tagged NUS-Wide Dataset). Our location encoder models the Earth as a continuous function, learning semantically rich, CLIP-aligned features that are suitable for geo-localization. Additionally, our location encoder architecture generalizes, making it suitable for use as a pre-trained GPS encoder to aid geo-aware neural architectures.

![ALT TEXT](/figures/method.png)

## Method

Similarly to OpenAI's CLIP, GeoCLIP is trained contrastively by matching Image-GPS pairs. By using the MP-16 dataset, composed of 4.7M Images taken across the globe, GeoCLIP learns distinctive visual features associated with different locations on earth.

_🚧 Repo Under Construction 🔨_

## 🗺️📍 Worldwide Image Geolocalization

![ALT TEXT](/figures/inference.png)

### Usage: GeoCLIP Inference

```python
import torch
from PIL import Image
from model.GeoCLIP import GeoCLIP

model = GeoCLIP()
model.eval()

image_file = "images/Kauai.png"

image = Image.open(image_file)

with torch.no_grad():
    top_pred_gps, top_pred_prob = model.predict(image, top_k=5)

print("Top 5 Predictions")
print("=================")
for i in range(5):
    lat, lon = top_pred_gps[i]
    print(f"Prediction {i+1}: ({lat:.6f}, {lon:.6f})")
    print(f"Probability: {top_pred_prob[i]:.6f}")
    print("")
```

## 🌐 Pre-Trained Location Encoder

In our paper, we show that once trained, our location encoder can assist other geo-aware neural architectures. Specifically, we explore our location encoder's ability to improve multi-class classification accuracy. We achieved state-of-the-art results on the Geo-Tagged NUS-Wide Dataset by concatenating GPS features from our pre-trained location encoder with an image's visual features. Additionally, we found that the GPS features learned by our location encoder, even without extra information, are effective for geo-aware image classification, achieving state-of-the-art performance in the GPS-only multi-class classification task on the same dataset.

![ALT TEXT](/figures/downstream-task.png)

### Usage: Pre-Trained Location Encoder

```python
import torch
import torch.nn as nn
from model.location_encoder import LocationEncoder

gps_encoder = LocationEncoder()
gps_encoder.load_state_dict(torch.load('model/weights/location_encoder_weights.pth'))

loc_data = torch.Tensor([[40.7128, -74.0060], [34.0522, -118.2437]])  # NYC and LA in lat, long
output = gps_encoder(loc_data)
print(output.shape) # (2, 512)
```

## Acknowledgments

This project incorporates code from Joshua M. Long's Random Fourier Features Pytorch. For the original source, visit [here](https://github.com/jmclong/random-fourier-features-pytorch).

## Citation

```
@article{cepeda2023geoclip,
  title={GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization},
  author={Vivanco, Vicente and Nayak, Gaurav Kumar and Shah, Mubarak},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```
