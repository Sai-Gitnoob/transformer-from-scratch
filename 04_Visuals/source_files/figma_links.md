# Source Files & Reference Links

A working list of tools for building the diagrams in `04_visuals/diagrams/`, plus reference visualizations worth studying before you draw your own. Everything below was checked against a live search rather than pulled from memory, so links should be current as of this writing — worth a quick click-through before relying on them, since community content moves around.

---

## Editable Diagramming Tools

### Figma / FigJam

- **[Architecture Diagram Components](https://www.figma.com/community/file/989635781221754599/architecture-diagram-components)** — A free, duplicable Figma/FigJam component library built specifically for system/architecture-style diagrams (boxes, containers, connectors, service icons). Good general-purpose base for the encoder/decoder block diagrams — not Transformer-specific, but the component set (frames, arrows, labeled containers) maps well onto encoder/decoder stacks.
  - Companion example: [Architecture Diagram Example — Multiplayer](https://www.figma.com/community/file/989634471195357925/architecture-diagram-example-multiplayer) shows the components used in a real diagram, useful as a layout reference.
  - Source/README on GitHub: [figma/architecture-diagram-components](https://github.com/figma/architecture-diagram-components)
- **[Technical Architecture Diagram Templates](https://www.figma.com/community/diagramming/technical-architecture)** — Figma's own curated collection of editable system-architecture templates. Broader than ML specifically, but useful for the container/stack visual language (matches the encoder/decoder "boxes inside boxes" structure well).
- **["3D Transformer" community file](https://www.figma.com/community/file/1285883829352903958/3d-transformer-community)** — A Figma community file specifically tagged as a 3D Transformer visualization. Worth a look for a more illustrative (less boxes-and-arrows) style, though it wasn't possible to verify its exact content/quality without opening it directly in Figma — check it in-app before committing to it as a base.

**Note:** none of the above are purpose-built "Attention Is All You Need" templates — Figma's community doesn't currently have a canonical one. The Architecture Diagram Components library is the most practical starting point for building the encoder/decoder stack diagrams from scratch.

### Excalidraw

- **[libraries.excalidraw.com](https://libraries.excalidraw.com)** — The official public library index. Hundreds of community-submitted icon/shape libraries (software architecture, system design, basic wireframing, etc.), all MIT-licensed and importable with one click. No dedicated "Transformer" or "attention mechanism" library currently exists here — the closest useful ones are the general **software architecture** and **system design** libraries, which give you labeled boxes, containers, and connectors similar to what you'd want for the encoder/decoder stack.
- Excalidraw's hand-drawn aesthetic is a nice fit for the "story" chapters (01_story) if you want the diagrams to feel more sketch-like and approachable than the more formal core-concepts diagrams.

### draw.io / diagrams.net

No pre-built "Transformer architecture" template turned up in search — draw.io doesn't currently have a dedicated ML/attention shape library the way it does for AWS or network diagrams. For this project, draw.io is realistically a **build-from-scratch** tool here: use its basic flowchart shapes (rectangles, containers, arrows) to replicate the block-diagram style. If a dedicated shape library shows up later, add it here.

---

## Reference Visualizations (for inspiration — not to copy directly)

These are well-known, widely-cited explainers of this exact paper. Great for checking your own diagram's accuracy and getting layout ideas — but treat them as references, not source material to trace or reproduce. Several are under specific licenses (noted below); redraw the *idea*, not the exact image.

- **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar.** The most widely cited visual walkthrough of this exact paper; referenced in AI/ML courses at MIT, Stanford, Harvard, and others. Step-by-step diagrams covering self-attention, multi-head attention, positional encoding, and the encoder-decoder structure — essentially the same arc as this book's `02_core_concepts` and `03_transformer` folders. **License: CC BY-NC-SA 4.0** — non-commercial, share-alike, attribution required if you adapt anything from it directly.
- **[The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) — Harvard NLP (Alexander Rush et al.).** A line-by-line PyTorch implementation of the paper, interleaved with the original text and diagrams. Less about pretty illustration, more about grounding every equation in actual working code — a good cross-reference for `05_code`. Source notebook: [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer).
- **[Transformer Explainer](https://poloclub.github.io/transformer-explainer/) — Georgia Tech Poloclub.** A fully interactive, in-browser visualization of a live GPT-2-scale Transformer, letting you type text and watch embeddings, attention, and token prediction happen in real time. The best reference if you want to build an *interactive* diagram for `04_visuals` rather than a static one.
- **[BertViz](https://github.com/jessevig/bertviz) — Jesse Vig.** An interactive Python tool (runs in Jupyter/Colab) for visualizing attention weights inside real trained models — head view, model view, and neuron view. Directly extends the original Tensor2Tensor visualization tool built by Llion Jones (one of this paper's actual co-authors). Most useful for `06_examples`, if you want to show real attention weights from an actual model rather than the illustrative numbers used in the core concepts chapters.

---

## A Note on Licensing

If any diagram in this project is adapted from Jay Alammar's work specifically, it needs attribution per his CC BY-NC-SA license — his suggested citation format:

> Alammar, J (2018). The Illustrated Transformer [Blog post]. Retrieved from https://jalammar.github.io/illustrated-transformer/

The Harvard NLP and Poloclub resources are both open-source/academic and generally fine to reference and link to directly; check each repo's license file before reusing actual code or image assets wholesale.