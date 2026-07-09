# Wave Mold Solver

Solver de modos guiados em guias de onda planares (slab waveguides), desenvolvido para análise de estruturas ópticas multicamada (ex: SOI, moldes fotônicos).

Dado um empilhamento de camadas (índices de refração e espessuras), o solver encontra os modos TE/TM guiados, calculando o índice efetivo (`neff`), a constante de propagação (`beta`) e o perfil de campo transversal de cada modo.

## Métodos

- **TMM (Transfer Matrix Method)** — implementado. Resolve a equação característica do guia varrendo o intervalo de índices efetivos possíveis e refinando as raízes com `scipy.optimize.brentq`. Suporta polarizações TE e TM.
- **FEM** — planejado, ainda não implementado (`solvers/fem.py`).

## Estrutura do projeto

```
config.json          # configuração padrão do problema (método, comprimento de onda, polarização, camadas)
main.py              # CLI: carrega o config, roda o solver, plota e imprime os modos
core/
  models.py          # dataclasses: Layer, Problem, ModeResult, FieldProfile
  materials.py        # relações físicas (kx, kappa, fatores de acoplamento TE/TM)
  postprocess.py      # reconstrução do perfil de campo e geração dos gráficos
solvers/
  tmm.py             # implementação do Transfer Matrix Method
  fem.py             # stub para o método de elementos finitos (futuro)
outputs/             # PNGs com os perfis de campo gerados
```

## Configuração

O empilhamento é definido em `config.json` como uma lista de camadas, sendo a primeira e a última semi-infinitas (`thickness_nm: null`):

```json
{
  "method": "TMM",
  "wavelength_nm": 1550,
  "polarization": "TE",
  "layers": [
    {"n": 1.44, "thickness_nm": null},
    {"n": 3.47, "thickness_nm": 220},
    {"n": 1.44, "thickness_nm": null}
  ]
}
```

## Uso

```bash
# usando o config.json padrão
python main.py

# sobrescrevendo parâmetros via linha de comando
python main.py --method TMM --wavelength 1550 --polarization TE --out-dir outputs

# usando outro arquivo de configuração
python main.py -c meu_config.json
```

Se nenhum argumento for passado, o CLI entra em modo interativo, perguntando método, polarização e comprimento de onda (Enter mantém o valor do `config.json`).

A saída lista os modos guiados encontrados:

```
Modo 0: neff = 3.176543, beta = 1.287654e+07 rad/m
Perfil de campo salvo em: outputs/modes_TE_1550nm.png
```

## Dependências

- numpy
- scipy
- matplotlib

```bash
pip install numpy scipy matplotlib
```
