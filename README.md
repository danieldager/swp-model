# single-word-processing-model

### Datasets:

### Architecture: 
Auditory pathway:
Sequence processing: start with RNNs, trained on a sequence of phones. 
Word repetition: for both comprehension and production. 

Visual pathway:
CNNs. Explore with standard CNNs, as in AlexNet, first trained on images of objects then refined on letter strings (see Hannagan et al., 2021). 
Test with CORnet

### Training:
For RNNs: Create a corpus with 50K words, including morphologically complex words. 
For each word have the phonological transcription (if needed, look for phonetic databases, text-to-speech TTS tools, etc). 
Create 

### Evaluation:
* Cross validation on left-out words from the lexicon. 
* Factorial Design: Sarah’s list of words based on the factorial design (see figure below). Convert to a list of.
* Pseudowords: Create a list of Pseudowords. Use Pallier’s pseudoword generator, unipseudo, reachable from openlexicon.fr.
* Real words:  Convert to strings of phones and images with written words.  


### Code:
Keep things modular. 
Prepare a class for each part of the model. For example, AuditoryEncoder, VisualEncoder, Decoder (production). 
Keep in mind that,  in the future, we may add working-memory mechanisms into units of the model.

### Citations:
Hannagan, T., Agrawal, A., Cohen, L., & Dehaene, A. S. (2021). Emergence of a compositional neural code for written words: Recycling of a convolutional neural network for reading. Proceedings of the National Academy of Sciences, 118(46), e2104779118.

Agrawal, A., & Dehaene, S. (2024). Cracking the neural code for word recognition in convolutional neural networks. arXiv preprint arXiv:2403.06159.

Agrawal, A., & Dehaene, S. (2023). Dissecting the neuronal mechanisms of invariant word recognition. bioRxiv, 2023-11.

Kubilius, J., Schrimpf, M., Nayebi, A., Bear, D., Yamins, D.L.K., DiCarlo, J.J. (2018) CORnet: Modeling the Neural Mechanisms of Core Object Recognition. biorxiv. doi:10.1101/408385

Kubilius, J., Schrimpf, M., Kar, K., Rajalingham, R., Hong, H., Majaj, N., ... & Dicarlo, J. (2019). Brain-like object recognition with high-performing shallow recurrent ANNs. In Advances in Neural Information Processing Systems (pp. 12785-12796).

Burgess, N., & Hitch, G. J. (1992). Toward a network model of the articulatory loop. Journal of memory and language, 31(4), 429-460.

Botvinick, M. M., & Plaut, D. C. (2006). Short-term memory for serial order: a recurrent neural network model. Psychological review, 113(2), 201.

Sajid, N., Holmes, E., Costa, L. D., Price, C., & Friston, K. (2022). A mixed generative model of auditory word repetition. bioRxiv, 2022-01.
