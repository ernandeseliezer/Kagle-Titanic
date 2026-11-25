# Titanic: Machine Learning from Disaster

Solução do desafio [Titanic do Kaggle](https://www.kaggle.com/competitions/titanic), uma competição de aprendizado para quem está começando em machine learning.

## O desafio

Em 1912, o RMS Titanic naufragou e nem todos os passageiros sobreviveram. O objetivo é usar dados dos passageiros (classe, sexo, idade, tarifa, família a bordo etc.) para treinar um modelo que **preveja quem sobreviveu**.

Os dados vêm divididos em dois arquivos:
- `train.csv`: passageiros com a resposta (`Survived`), usados para treinar o modelo.
- `test.csv`: passageiros sem a resposta, para os quais o modelo faz as previsões.

## O que foi feito

1. **Exploração dos dados:** nulos, contagens e gráficos de sobrevivência por categoria.
2. **Tratamento de nulos:** `Age` preenchida pela média da classe do passageiro; `Embarked` e `Fare` com poucos nulos.
3. **Novas features:** `Title` (título extraído do nome), `FamilySize` (tamanho da família) e `HasCabin` (se tinha cabine registrada).
4. **Conversão para números:** `Sex` em 0/1 e `get_dummies` em `Pclass`, `Title` e `Embarked`.
5. **Modelos:** Random Forest e Gradient Boosting, comparados com validação cruzada, e geração do arquivo de previsões.

## Arquivos

| Arquivo | Descrição |
|---|---|
| `titanic.ipynb` | Notebook com toda a análise |
| `train.csv`, `test.csv` | Dados da competição |
| `gender_submission.csv` | Exemplo de envio fornecido pelo Kaggle |
| `titanic_gradient_boosting_pred.csv` | Previsões geradas pelo modelo |

## Como rodar

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook
```

Abra o `titanic.ipynb` e rode todas as células.

## Tecnologias

Python, Pandas, NumPy, Matplotlib, Seaborn e Scikit-learn.
