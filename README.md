# forabot

A Keras implementation for foraminifera identification

# Install

### MacOSX, Linux, Windows

```bash
docker build --tag forabot .
docker run -it forabot /bin/bash
```

### Jetson Nano 2GB

```bash
# pull image from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorflow
sudo docker pull nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3
# mount code and ssh into container
sudo docker run -it --volume=/path/to/repo:/repo --rm --runtime nvidia --network host nvcr.io/nvidia/l4t-tensorflow:r32.6.1-tf2.5-py3
```

# References

B. Zhong, Q. Ge, B. Kanakiya, R. Mitra, T. Marchitto, E. Lobaton, “A Comparative Study of Image Classification Algorithms for Foraminifera Identification,” IEEE Symp. Series on Computational Intelligence (SSCI), 2017. [Makes use of NCSU-CUB Foram Images 01 Dataset]

R. Mitra, T.M. Marchitto, Q. Ge, B. Zhong, B. Kanakiya, M.S. Cook, J.S. Fehrenbacher, J.D. Ortiz, A. Tripati, E. Lobaton, “Automated species-level identification of planktic foraminifera using convolutional neural networks, with comparison to human performance,” Marine Micropaleontology, 147, 2019, pp 16-24.
