# Trabalho Final - Inteligência Artificial

## Sistema Inteligente de Triagem em Pronto-Socorro

### Integrantes:

- João Pedro Gondim Marreira - 539230
- Jonathan Davi Sampaio Faria - 496357

## Instruções:

Para executar os experimentos:

```bash
python3 src/test.py
```

Para utilizar a API:

```python
t = TriageManager(consultation_time=20)
order = t.order([ Report(...), ... ])
risk = t.sn.risk_acc(order)
```
