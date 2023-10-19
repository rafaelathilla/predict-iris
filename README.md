## Introdução 
Este projeto tem por finalizade a contrução de um modelo de classificação e a disponibilização das predições por meio de uma API para classificação de plantas do tipo Iris.

## O projeto pode ser iniciado de 2 formas:

1. Ambiente Local
   
   Faça o clone do projeto e acesse o diretório raiz:
   ```
   git clone https://github.com/rafaelathilla/predict-iris.git
   ```
   Crie o ambiente virtual Python (Versão utilizada é a 3.8):
   ```
   python3.8 -m venv env
   ```
   Criação o ambiente virtual:
   ```
   Windows:  .\env\Scripts\activate
   Linux:    source /env/bin/activate  
   ```
   Instação das bibliotecas necessárias:
   ```
   pip install -r requirements.txt
   ```
   Para iniciar a API:
   ```
   python main.py
   ```

2. Utilizando Docker
   
   Uma imagem com o projeto já foi disponibilizada em um repositório publico, o container pode ser iniciado da seguinte forma:
   ```
   docker run -d -p 8001:8001 rafaelathilla/iris-predict
   ```

## Reprodução da Pipeline para Criação, treinamento, avaliação e disponibilização do modelo.

   Partindo o princípio de que o clone do projeto foi realizado, ambiente virtual criado e ativado, a pipeline pode ser iniciada a partir do comando:
   ```
   python ./train_model.py
   ```

   Os dados necessários serão carregados, o pré-processamento será feito, o modelo será treinado e avaliado.

   Ao final da execução a console irá apresentar os as metricas do modelo, bem como a matriz de confusão.

    













