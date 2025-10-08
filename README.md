<div align="center">    
 
# 🌎 GeoCLIP-PLUS: Multi-Modal Geo-localization

[![Paper](http://img.shields.io/badge/paper-arxiv.2309.16020-B31B1B.svg)](https://arxiv.org/abs/2309.16020v2)
[![Conference](https://img.shields.io/badge/NeurIPS-2023-blue)]()

![ALT TEXT](/figures/GeoCLIP.png)
## Acknowledgments

This project is  based on the work of Vicente Vivanco Cepeda. It also incorporates code from Joshua M. Long's Random Fourier Features Pytorch. For the original source, visit [here](https://github.com/jmclong/random-fourier-features-pytorch).
</div>

### 📍 Try The Demo! [![Colab Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/*pending*)

## Description

GeoCLIP-Plus builds on the original GeoCLIP model by incorporating advanced features that enhance usability and expand functionality. This version introduces more user-friendly interfaces and additional utilities, designed to streamline the workflow for geospatial data analysis and visualization.

![ALT TEXT](/figures/method.png)

## Method

Similarly to OpenAI's CLIP, GeoCLIP is trained contrastively by matching Image-GPS pairs. By using the MP-16 dataset, composed of 4.7M Images taken across the globe, GeoCLIP learns distinctive visual features associated with different locations on earth.


## 📎 Getting Started:

Install directly from source:

```
git clone https://github.com/LangGang-AI/geo-clip-plus
cd geo-clip-plus
python setup.py install
```

## 🗺️📍 Image Geolocalization

![ALT TEXT](/figures/inference.png)

### Usage: GeoCLIP Inference

```python
import torch
from geoclip import GeoCLIP

model = GeoCLIP()

image_path = "image.png"

top_pred_gps, top_pred_prob = model.predict(image_path, top_k=5)

print("Top 5 GPS Predictions")
print("=====================")
for i in range(5):
    lat, lon = top_pred_gps[i]
    print(f"Prediction {i+1}: ({lat:.6f}, {lon:.6f})")
    print(f"Probability: {top_pred_prob[i]:.6f}")
    print("")
```

## Text Localization 
'''python 


## 🌐 Worldwide GPS Embeddings


![ALT TEXT](/figures/downstream-task.png)

### QuickStart: Get Location Embeddings with the Pre-trained Encoder

```python
import torch
from geoclip import LocationEncoder

gps_encoder = LocationEncoder()

gps_data = torch.Tensor([[40.7128, -74.0060], [34.0522, -118.2437]])  # NYC and LA in lat, lon
gps_embeddings = gps_encoder(gps_data)
print(gps_embeddings.shape) # (2, 512)
```

## Acknowledgments

This project is  based on the work of Vicente Vivanco Cepeda. It also incorporates code from Joshua M. Long's Random Fourier Features Pytorch. For the original source, visit [here](https://github.com/jmclong/random-fourier-features-pytorch).

## Citation

If you find GeoCLIP or GeoCLIP-Plusbeneficial for your research, please consider citing us with the following BibTeX entry:

```
@inproceedings{geoclip,
  title={GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization},
  author={Vivanco, Vicente and Nayak, Gaurav Kumar and Shah, Mubarak},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
