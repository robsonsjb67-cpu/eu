# eu — leitura visual via OBS

Pipeline de **percepção** (somente leitura) que observa o jogo pela saída de vídeo do OBS.
Nenhum módulo envia teclas, cliques ou altera o foco de janelas.

| Arquivo | Função |
|---|---|
| `obs_capture.py` | Captura independente (OBS Virtual Camera, stream SRT/UDP ou arquivo), fila de frames recentes, detecção de desconexão e de frames congelados, reconexão automática. |
| `battle_attack.py` | Leitor da Battle List: barras de vida → linhas, % de vida, cor, OCR opcional do nome, moldura de destaque, rastreador com IDs temporários (entrada/saída/mudança de vida, sem contagem duplicada). |
| `monster_detector.py` | Biblioteca de templates por monstro, `matchTemplate` com máscara alfa, limiar global/por monstro, NMS para sobreposição, rastreamento temporal para animações e efeitos. |
| `target_fusion.py` | Fusão Battle List + imagem: status confirmado / provável / incerto / saiu, evidências e histórico por alvo, IDs estáveis, sem duplicatas. |
| `main.py` | CLI: selecionar regiões, cadastrar templates, modo de inspeção com overlay. |

## Preparação

1. No OBS, crie uma cena com a captura do jogo e clique em **Iniciar Câmera Virtual**
   (ou configure uma saída de stream local e use a URL em `capture.source`).
2. `pip install -r requirements.txt` (para OCR, instale também o binário do Tesseract).

## Uso

```bash
python main.py select-region battle_list     # arraste sobre a Battle List
python main.py select-region game_area       # arraste sobre a área de jogo
python main.py add-template "Rat"            # recorte o sprite no frame atual
python main.py add-template "Rat" --image rat_andando.png --threshold 0.85
python main.py run                           # overlay + tabela de alvos
```

No modo `run`: **i** imprime a explicação de cada alvo, **p** pausa a análise, **q** sai.

Dicas para templates: recorte justo ao sprite, cadastre vários frames da animação e direções,
e use PNG com transparência para ignorar o chão. Recortes parciais (ex.: só a metade superior)
ajudam quando a criatura costuma ficar parcialmente coberta.

## Configuração (`config.json`)

Criado ao selecionar regiões. Valores padrão em `vision_common.DEFAULT_CONFIG`; os principais:

- `capture.source`: índice da câmera virtual ou URL; `freeze_seconds` / `freeze_threshold` para congelamento;
  `disconnect_timeout` para desconexão.
- `battle_list.bar_full_width`: largura da barra com 100% de vida (se omitido, é aprendida).
- `detector.threshold`: limiar de confiança (0–1); por monstro via `templates/<nome>/meta.json`.
- `fusion.time_window`: janela de tempo (s) para associar eventos.

## Testes

```bash
pip install pytest && python -m pytest -q
```
