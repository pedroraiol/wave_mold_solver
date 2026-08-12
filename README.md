# Wave Mold Solver

Solver de modos guiados em guias de onda planares (slab waveguides), desenvolvido para análise de estruturas ópticas multicamada (ex: SOI, moldes fotônicos).

Dado um empilhamento de camadas (índices de refração e espessuras), o solver encontra os modos TE/TM guiados, calculando o índice efetivo (`neff`), a constante de propagação (`beta`) e o perfil de campo transversal de cada modo.

## Métodos

- **TMM (Transfer Matrix Method)** — implementado. Resolve a equação característica do guia varrendo o intervalo de índices efetivos possíveis e refinando as raízes com `scipy.optimize.brentq`. Suporta polarizações TE e TM.
- **FEM (elementos finitos, 1D)** — implementado. Discretiza o mesmo empilhamento de camadas com elementos lineares (P1), truncando as claddings semi-infinitas em uma caixa finita (padding proporcional ao comprimento de onda) com condição de contorno de Dirichlet (`ψ=0` nas bordas). Resolve o problema de autovalor generalizado resultante com `scipy.linalg.eigh` e filtra os modos guiados pela mesma janela de índice efetivo usada no TMM. Serve de validação cruzada do TMM e, por não depender de solução analítica por camada, abre caminho para perfis de índice graduais no futuro. Um FEM 2D escalar (para guias rib/strip) fica como trabalho futuro — exigiria um novo modelo de problema (malha 2D).

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
  fem.py             # implementação do método de elementos finitos (1D)
outputs/             # PNGs com os perfis de campo gerados
tests/               # suíte de testes (pytest)
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

## Testes

A suíte cobre as funções físicas de `core/materials.py`, os dois solvers
individualmente (contagem de modos validada analiticamente via V-number de
slab simétrico, ordenação, bounds, caso sem modo guiado, polarização
inválida) e uma validação cruzada TMM vs FEM (os dois métodos devem
concordar em `neff` dentro de ~4e-3, margem esperada do truncamento de
domínio do FEM).

```bash
pip install pytest
pytest
```

O FEM usa autovalor denso (`scipy.linalg.eigh`), então a suíte completa leva
em torno de 1 minuto — a maior parte é gasta nos casos parametrizados de FEM.

## Dependências

- numpy
- scipy
- matplotlib

```bash
pip install numpy scipy matplotlib
```
