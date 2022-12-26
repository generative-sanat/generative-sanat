# Text Embedding

Onceki bolumlerde inceledigimiz gibi diffusion modelimizi egitirken prompt ile istedigimzi ciktiyi uretmesi icin bir yazi ile gorsellerin latent representasyonlarini kondisyonluyorduk (conditioning) yani goruntuyu VAE'den gecirdikten sonra gorselin caption'ini da text embeddingden gecirip bu ikisini birlestiriyorduk.

Tabi bu captionlar yazi formatinda veriler oldugu icin bunlari temsil edecek sayilar formlara da ihtiyacimiz var bu isleme text embedding deniyor.  

Goruntuler ve onlarin captionlari ile calistigimiz icin CLIP text encoder'i kullanmak mantikli olacaktir. Cok kisaca CLIP encoder, daha onceki bolumlerden hatirlayacaginiz gibi goruntuler ve ve goruntulerin captionlari ile OpenAI tarafindan egitilmis open source bir modeldi.

## CLIP

Clip nedir?

## Uygulama

```python
text_encoder.text_model.embeddings

# CLIPTextEmbeddings(
#   (token_embedding): Embedding(49408, 768)
#   (position_embedding): Embedding(77, 768)
# )
```

```python
# Our text prompt
prompt = 'A picture of a puppy'
```

Tokenizer

```python
# Turn the text into a sequnce of tokens:
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
text_input['input_ids'][0] # View the tokens

# tensor([49406,   320,  1674,   539,   320,  6829, 49407, 49407, 49407, 49407,
#         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
#         49407, 49407, 49407, 49407, 49407, 49407, 49407])
```

```python
# See the individual tokens
for t in text_input['input_ids'][0][:8]: # We'll just look at the first 7 to save you from a wall of '<|endoftext|>'
    print(t, tokenizer.decoder.get(int(t)))
```

```python
# Grab the output embeddings
output_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
print('Shape:', output_embeddings.shape)
output_embeddings

# Shape: torch.Size([1, 77, 768])
# tensor([[[-0.3884,  0.0229, -0.0522,  ..., -0.4899, -0.3066,  0.0675],
#          [ 0.0290, -1.3258,  0.3085,  ..., -0.5257,  0.9768,  0.6652],
#          [ 0.6942,  0.3538,  1.0991,  ..., -1.5716, -1.2643, -0.0121],
#          ...,
#          [-0.0221, -0.0053, -0.0089,  ..., -0.7303, -1.3830, -0.3011],
#          [-0.0062, -0.0246,  0.0065,  ..., -0.7326, -1.3745, -0.2953],
#          [-0.0536,  0.0269,  0.0444,  ..., -0.7159, -1.3634, -0.3075]]],
#        device='cuda:0', grad_fn=<NativeLayerNormBackward0>)
```
