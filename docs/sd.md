# Diffusion Models

Temel notlar:

- Yuksek cozunurluklu goruntu islemek oldukca masraflidir cunku 128px bir goruntu 64px bir goruntunun tam 4 katidir bu baglamda yapilacak islemler quadratictir.
- Diffsuion modellerinde VAE (variational auto encoderlar) kullanilir, bu goruntunun ortulu bir uzayda daha kucuk bir representasyonunu saglar. 512x512x3 olan bir gorsel 64x64x4 boyutuna indirgenir.
- VAE ile birlikte gorsellerdeki redundant kisimlari atilmis olur, yeteri kadar data verildigi muddet VAE gorselleri latent spaceden tekrardan eski cozunurluge olusturabilmeyi ogrenir.
- Text Conditioning, diffusion modeli egitilirken goruntuye ek olarak bu goruntunun olusmasinda etkili olacak sekilde text datasinin da (caption) bilgi olarak verilmesine denir. Burda amaca gurultulu bir gorsel verildiginde modelin caption’a uygun bir sekilde gurultuyu cozmesi ve goruntunun buna gore olusmasini amaclar.
- Inference aninda, baslangicta pure noise ve olusturmak istedigimiz goruntuye uygun olacak sekilde bir text veririz ve modelin random bir inputu text’e gore olusturmasini isteriz.
- Text conditioning olusturmak icin yazilarin numerik bir representasyonunu olusturmamiz gerekiyor, bunu CLIP adi verilen OpenAI tarafindan gelistirilmis bir LLM modeli kullaniyoruz. CLIP, image captionlari uzerinde egitilmis, fotograflari ve yazilarini (fotograflarin captionlarini) karsislastirmamizi saglayan bir model.
- Stable diffusionda kullandigimiz promptlar once clip encoder'a iletilir ve SD 1.x serisi icin token basina 768 uzunlugunda bir embedding vektoru uretir. Bu SD 2.x versiyonlari icin 1024 uzunlugunda bir vektor.
- Sureci stabil tutmak adina tum promptlar 77 token ile sinirlandirilmistir. Yani daha uzun prompt girmeniz tokenlerin truncate edilmesine sebep olur. Son durumda CLIP'den elde ettigimiz matrix 77x768 uzunlugunda olacaktir. SD 2.x icin 77x1024.

![Conditioning](https://github.com/huggingface/diffusion-models-class/blob/main/unit3/sd_unet_color.png?raw=true)

- Cross Attention nedir?
- Her ne kadar text conditioning yapsak da, gun sonunda olusan gurultu baslangicta kullandigimiz noise inputa cok bagimli oldugunu goruyoruz. Bu mantikli cunku milyonlarca resimin caption'u cogunlukla resimin kendisinden bagimsiz captionlar iceriyor. Bu yuzden model descriptionlardan cok ogrenemiyor.
- Bu problemi ortadan kaldirmak icin CFG (Classifier Free Guidance) denilen bir yontem uyguluyoruz.
- Cok kisa bir tabirle model egitim sirasinda text bilgisi olmadan egitilir, inference aninda ise iki tahmin yapilir zero conditioning ve text conditioning, bu ikisi arasindaki farka bakarak bilr olcek olustururuz bu da CFG adi verilir.
- Super-resolution, Inpainting ve Depth to Image turunde 3 farkli conditioning de vardir, ayni textte oldugu gibi, super resolution fotografin yuksek cozunurluklu hali ve dusuk cozunurlukle hali uzerinde bir egitim gerceklestirilir, depth 2 image, midas modeli ile goruntunun kendisi ve derinlik haritasi cikartilmis haliyle conditionlanarak egitilir.
