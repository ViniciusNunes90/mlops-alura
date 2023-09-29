from doctest import debug
from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import pickle
import os

modelo = pickle.load(open('models/modelo.sav', 'rb'))
colunas = ['tamanho', 'ano', 'garagem']

# criação do app
app = Flask('__name__')

# Criação da autenticação e dados de acesso
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

# definição das rotas da api
@app.route('/')
def home():
    return "Minha primeira API."

@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt_br', to='en')
    polaridade = tb_en.sentiment.polarity

    return "polaridade: {}".format(polaridade)

@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    
    return jsonify(preco=preco[0])

app.run(debug=True, host='0.0.0.0')

# Gerar o arquivo de requirements: "pip freeze > requirements.txt"
# Instalar as lib's a partir do arquivo de requirements: "pip install -r requirements.txt"