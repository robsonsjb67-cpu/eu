# eu — análise visual do Tibia via OBS Studio

Sistema modular em Python que **observa** o Tibia pela saída de vídeo do OBS Studio:
reconhece a posição pelo minimapa e acompanha rotas (CaveBot), lê a Battle List,
detecta monstros na tela, estima o HP do personagem e de um aliado e controla
temporizadores de magias.

> **Somente leitura.** Nenhum módulo envia teclas, cliques, movimenta o personagem ou
> muda o foco da janela do jogo. Os módulos mostram o que foi reconhecido, registram
> eventos e emitem alertas visuais/sonoros (ex.: "HP 40% — Exura Vita recomendado").
> Leituras incertas ou desatualizadas nunca geram alerta nem recomendação.

## Módulos

| Arquivo | Função |
|---|---|
| `main.py` | Ponto de entrada: interface gráfica (padrão) e comandos de linha (observação, análise de gravações). |
| `interface.py` | Interface PySide6: prévia do OBS, seleção de regiões com o mouse, editor de rotas, estado do CaveBot, Battle List, HP/SIO, temporizadores, logs, alertas e parada de emergência (F12). |
| `obs_capture.py` | Captura em thread própria (OBS Virtual Camera, stream SRT/UDP, vídeo ou pasta de imagens gravadas), prints/gravação de frames, fila dos frames mais recentes, detecção de desconexão e de frames congelados, reconexão automática. |
| `analysis.py` | Motor que entrega cada frame aos módulos; cada módulo roda isolado (um erro não derruba os outros). |
| `route_manager.py` | Rotas e waypoints: criar, editar, ordenar, salvar/carregar/renomear/excluir em JSON, validação. |
| `cave_navigation.py` | Referências do minimapa, estimativa de posição com limiar e checagem de ambiguidade, perda de referência, suspeita de troca de andar, calibração da região. |
| `cavebot.py` | Acompanha a rota: waypoint atual/próximo/progresso, iniciar/pausar/continuar/parar, conclusão só por reconhecimento visual, pausa automática sem posição confiável, registro de falhas, modo de observação com gravações. |
| `battle_attack.py` | Battle List: linhas, barras de vida, nomes (OCR opcional), entradas/saídas, sem contagem duplicada. |
| `monster_detector.py` | Detecção de monstros na área de jogo por referências visuais (templates com máscara, NMS, rastreamento temporal). |
| `target_fusion.py` | Compara Battle List × detecção visual: confirmado / provável / incerto / saiu, com evidências e confiança. |
| `health_monitor.py` | Barra de HP: % estimada com OpenCV, confiança, leituras desatualizadas, limite configurável com histerese, eventos e alertas (Exura Vita). |
| `sio_monitor.py` | Barra de um aliado: leitura com confiança e identidade **não presumida** (só é "confirmada" com imagem de referência do nome). |
| `spell_timers.py` | Temporizadores: Utani Gran Hur (30 s) e Utamo Vita (50 s) por padrão; iniciar, pausar, continuar, reiniciar, aviso antes do fim e registro. |
| `config.py` | Configuração persistente (`config.json`) com padrões e gravação atômica. |
| `logger.py` | Log em arquivo rotativo (`logs/eu.log`) e histórico de eventos exibido na interface. |
| `vision_common.py` | Regiões, NMS e utilidades compartilhadas. |

## Instalação no Windows

1. Instale o **Python 3.11 ou superior** em <https://www.python.org/downloads/>
   (marque **"Add python.exe to PATH"** no instalador).
2. Instale o **OBS Studio** (28 ou superior) em <https://obsproject.com/>.
3. Baixe/clone este projeto, abra o **PowerShell** na pasta e rode:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   Se o PowerShell bloquear o `Activate.ps1`, rode uma vez
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (ou use o `cmd` com `.venv\Scripts\activate.bat`).

4. (Opcional) Para ler os nomes da Battle List por OCR, instale o
   [Tesseract para Windows](https://github.com/UB-Mannheim/tesseract/wiki) e adicione
   `C:\Program Files\Tesseract-OCR` ao `PATH`. Sem ele, o resto funciona normalmente.

## Preparando o OBS

1. Crie uma cena com uma fonte **Captura de jogo** ou **Captura de janela** do cliente do Tibia.
   Mantenha a resolução da cena igual à do jogo e **não redimensione** depois de calibrar.
2. Clique em **Iniciar Câmera Virtual** (painel Controles). A câmera virtual é a fonte `0`
   (ou `1`, `2`… se houver webcams).
   - Alternativa: em *Configurações → Transmissão/Gravação* use uma saída local (ex.: SRT)
     e informe a URL (`srt://127.0.0.1:9000`) no campo **Fonte OBS**.
3. A captura lê apenas o vídeo do OBS: o Tibia pode ficar em segundo plano ou em outro monitor.

### Salvar prints e gravar frames

- **Print** (ou **F9**): salva o frame atual em PNG na pasta `prints/`.
- **Gravar frames**: salva uma sequência em `prints/gravacao_<data>/` no ritmo de
  `capture.playback_fps`. Essa pasta pode ser usada em **CaveBot → Testar com gravação…**,
  em `python main.py observe --recording ...` ou como **Fonte OBS** para reproduzir a sessão.

## Execução

```powershell
.\.venv\Scripts\Activate.ps1
python main.py
```

Fluxo sugerido na interface:

1. **Fonte OBS** → `0` → **Conectar**. A prévia mostra o vídeo, o FPS e o estado
   (conectada / congelada / desconectada).
2. Aba **Regiões**: clique em *Selecionar Minimapa* e arraste sobre o minimapa na prévia;
   repita para Battle List, Área de jogo, Barra de HP e Barra do aliado. Use
   **Verificar calibração do minimapa** (centralize a região na cruz branca do personagem).
3. Aba **CaveBot**: **Nova** rota → **Adicionar** waypoints (nome, tipo, raio e, se souber,
   coordenadas). Com o personagem **parado em cada waypoint**, selecione-o e clique em
   **Usar minimapa atual como referência**. Cadastre referências a cada ~8–10 sqm
   (o reconhecimento só funciona perto de uma referência).
   **Iniciar** acompanha a rota; a tabela destaca o waypoint atual.
4. Aba **Battle**: **Cadastrar referência visual de monstro** e recorte o sprite na prévia
   (vários frames/direções melhoram a detecção).
5. Aba **Cura / Suporte**: ajuste o HP mínimo; com HP cheio, **Calibrar 100%**. Para o
   SIO, **Cadastrar nome do aliado** recortando o nome dele na prévia.
6. Aba **Magias**: use **Iniciar** ao lançar a magia; ajuste o intervalo ao lado.
7. **PARADA DE EMERGÊNCIA** (ou **F12**) interrompe toda a análise, para o CaveBot e zera
   os temporizadores; **Liberar análise** retoma.

Tudo (regiões, limiares, temporizadores, última rota, tamanho da janela) fica em `config.json`.
Rotas ficam em `routes/*.json`, referências do minimapa em `minimap_refs/`,
referências de monstros em `templates/` e o log em `logs/eu.log`.

### Como o CaveBot decide

- A posição só é estimada quando a similaridade com uma referência passa do limiar
  (`cavebot.match_threshold`) **e** não há outra referência quase igual apontando para
  outro lugar (`ambiguity_margin`).
- Coordenadas só aparecem se a referência tiver coordenadas; caso contrário, a posição é
  relativa à referência (ex.: "deslocado +6, −4 px"). Nada é inventado.
- Um waypoint só é concluído depois de reconhecido em `confirm_frames` frames seguidos —
  **nunca por tempo**.
- Sem posição confiável por `lost_pause_seconds`, o estado vira **aguardando posição** e
  o motivo aparece em *Falhas e motivos*. Mudanças bruscas do minimapa sem explicação por
  deslocamento são marcadas como possível troca de andar.
- `cavebot.pixels_per_sqm` converte pixels do minimapa em sqm e depende do zoom do minimapa e
  da escala do OBS. Para calibrar: ande N sqm em linha reta e veja o deslocamento em pixels
  exibido em *Posição*; `pixels_per_sqm = pixels / N`.

## Modo de observação (testar rotas com gravações)

Grave uma sessão no OBS (*Iniciar gravação*) ou salve frames em uma pasta (PNG/JPG) e:

- na interface: **CaveBot → Testar com gravação…**; ou
- na linha de comando:

  ```powershell
  python main.py observe --route "Rotworm Cave" --recording C:\gravacoes\sessao.mp4 --fps 30 --step 3
  python main.py analyze --recording C:\gravacoes\sessao.mp4 --verbose   # HP/SIO/Battle
  ```

O relatório lista os waypoints reconhecidos (com instante e motivo) e as falhas
(referência perdida, ambiguidade, waypoint fora de ordem, andar inesperado), para você
recalibrar a rota. Também é possível usar uma gravação como **Fonte OBS** (botão
**Abrir gravação…**) e ver tudo na interface como se fosse ao vivo.

## Testes

```powershell
pip install pytest
python -m pytest -q
```

Os testes geram imagens sintéticas no estilo do jogo (minimapa com cruz do personagem,
barras de HP, Battle List, sprites), gravam frames em disco e os reproduzem pelo mesmo
caminho da captura real. Inclui um teste da interface em modo `offscreen`.

## Solução de problemas

| Sintoma | O que verificar |
|---|---|
| "captura: DESCONECTADA" | Câmera Virtual do OBS iniciada? Índice correto (`0`, `1`…)? Outro programa usando a câmera virtual? |
| "captura: CONGELADA" | A fonte do OBS travou (jogo minimizado com *Captura de jogo*?). Ajuste `capture.freeze_seconds`. |
| Minimapa "sem detalhe" | Região fora do minimapa, ou OBS redimensionado depois da calibração. |
| CaveBot sempre "confiança baixa" | Cadastre referências mais próximas; confira `pixels_per_sqm`; reduza `match_threshold` com cuidado. |
| HP com "?" | Leitura incerta (confiança abaixo do mínimo). Recalibre a região/100% ou ajuste `health.fill_hsv_ranges`. |
