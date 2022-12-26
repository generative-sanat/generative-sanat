# Glossary

## Noise

## UNet

- GroupNorm applies group normalization to the inputs of each block
- Dropout layers for smoother training
- Multiple resnet layers per block (if layers_per_block isn't set to 1)
- Attention (usually used only at lower resolution blocks)
- Conditioning on the timestep.
- Downsampling and upsampling blocks with learnable parameters

## fp16 (half precision)

In computing, half precision (sometimes called FP16) is a binary floating-point computer number format that occupies 16 bits (two bytes in modern computers) in computer memory. It is intended for storage of floating-point values in applications where higher precision is not essential, in particular image processing and neural networks.

![fp16](https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/IEEE_754r_Half_Floating_Point_Format.svg/2880px-IEEE_754r_Half_Floating_Point_Format.svg.png)

## DDPM

## Schedulers or Samplers

# Sources

- <https://github.com/huggingface/diffusion-models-class/tree/main/unit1>
