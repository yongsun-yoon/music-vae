# music-vae

## 개요
이 저장소는 Music VAE 모델을 사용해 4마디의 드럼 샘플을 생성하기 위해 개발되었습니다.

Music VAE는 논문 [A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music](https://arxiv.org/abs/1803.05428)에서 제안한 모델입니다. 논문에 대한 자세한 내용은 아래 [논문 리뷰]에 작성했습니다.

모든 코드는 [공식 저장소](https://github.com/magenta/magenta/tree/main/magenta/models/music_vae)에서 공개한 코드를 기반으로 합니다.



## 실행 방법
실행은 총 세 단계로 구성됩니다. 각 단계에 따라 colab notebook을 실행하면 됩니다.

1. [01_preprocess.ipynb](01_preprocess.ipynb): Groove MIDI 데이터셋을 다운로드 후 전처리합니다.
2. [02_train.ipynb](02_train.ipynb): Music VAE 모델을 학습합니다.
3. [03_generate.ipynb](03_generate.ipynb): 학습한 Music VAE 모델로 드럼 샘플을 생성합니다.


## 결과
생성 샘플은 [samples](samples) 디렉토리에서 확인 가능합니다.
* public: 저자가 공개한 모델([groovae_4bar](https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/groovae_4bar.tar))을 사용하여 생성
* private: 02_train 단계에서 직접 학습한 모델로 생성


## 논문 리뷰
Music VAE는 Variational Autoencoder(VAE)를 사용한 음악 생성 모델입니다. 

<br />

VAE는 latent variable 기반의 생성 모델로서 고차원 데이터의 특징을 샘플링 가능한 저차원의 latent vector로 표현하여 다양한 데이터를 생성할 수 있습니다. 또한 VAE는 encoder를 통해 $p(z|x)$를 모델링함으로써 데이터의 attribute를 추출하고 조작할 수 있는 장점이 있습니다.

<br />

하지만 VAE를 sequential data에 적용할 경우 posterior collapse 문제가 발생할 수 있습니다. Autoregressive decoder가 latent vector가 아닌 autoregressive한 능력으로만 데이터를 생성하기 때문에, high quality sampling과 reconstruction이 안되는 문제입니다.

<br />

Posterior collapse 문제를 해결하기 위해 ELBO(Evidence Lower BOund)에서 regularization term의 비중을 약화시키는 연구들이 제안되었습니다. 
1보다 작은 가중치 $\beta$를 regularization term에 곱하거나, threshold $\tau$ 보다 큰 경우에만 regularization loss를 적용하는 방법들이 있습니다.

<br />

Music VAE는 이러한 연구들에 덧붙여 hierarchical decoder를 사용함으로써 posterior collapse를 해결하고자 했습니다. Encoder가 latent vector를 만들면 conductor 레이어를 통해 segment hidden state를 만듭니다. segment는 전체 sequence를 겹치지 않게 나눈 subsequence 입니다. 즉, conductor는 생성할 sequence의 전반적인 흐름을 짜는 역할을 합니다.

<br />

각각의 segment state는 또 다른 LSTM의 initial state가 됩니다. Conductor가 segment를 만들고, segment마다 LSTM이 있기 때문에 hierarchical 구조가 됩니다. 이러한 구조의 모델을 사용함으로써 posterior collapse를 해결하고 long-sequence에 대해 high quality sample을 생성하였습니다.
