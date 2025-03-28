# flow-generator-matching-distillation


Learning flow matching models is easy. Mak- ing them fast and practical is a tough task and a hot research topic. We propose a framework for the distillation and acceleration of genera- tive models by integrating techniques from Flow Generator Matching and Inverse Bridge Matching Distillation. Flow matching models have set the state-of-the-art in generating high-fidelity sam- ples, but their substantial sampling computational demands limit their practical deployment. Our work involves training a base generative model, followed by a distillation pipeline that compresses the large model into compact, efficient one-step and multi-step generators while maintaining per- formance. This pipeline leverages theoretical guarantees from recent advances in flow matching and is evaluated through extensive experiments comparing the distilled models with its original counterpart. Results demonstrate significant im- provements in inference speed and resource effi- ciency with minimal loss in sample quality, un- derscoring the potential of flow matching-based distillation for practical generative modeling.


Flow Generator Matching (FGM), an innovative approach designed to accelerate the sampling of flow-matching models into a one-step generation.
