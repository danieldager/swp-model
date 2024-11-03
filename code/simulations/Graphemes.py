import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, TensorDataset

# refactored from @author: aakash
# NOTE: Missing image transformations and perturbations 
def text_to_grapheme(
    self, words: list=["text"], savepath=None, index=1, 
    fontname='Arial', W = 64, H = 64, size=10, spacing=0,
    xshift=0, yshift=-3, upper=False, invert=False, mirror=False, show=None
) -> list:

    tensors = []
    for word in words:
        if upper: word = word.upper()
        if invert: word = word[::-1]
        
        img = Image.new("L", (W,H), color=10)
        fnt = ImageFont.truetype(fontname+'.ttf', size)
        draw = ImageDraw.Draw(img)

        # Starting word anchor
        w = sum([(fnt.getbbox(l)[2] - fnt.getbbox(l)[0]) for l in word])
        h = sum([(fnt.getbbox(l)[3] - fnt.getbbox(l)[1]) for l in word]) / len(word)
        w = w + spacing * (len(word) - 1)
        h_anchor = (W - w) / 2
        v_anchor = (H - h) / 2

        x, y = (xshift + h_anchor, yshift + v_anchor)
        
        # Draw the word letter by letter
        for l in word:
            draw.text((x,y), l, font=fnt, fill="white")
            letter_w = fnt.getbbox(l)[2] - fnt.getbbox(l)[0]
            x += letter_w + spacing

        if x > (W + spacing + 2) or (xshift + h_anchor) < -1:
            raise ValueError(f"Text width is bigger than image. Failed on size:{size}")
        
        if savepath:
            img.save(f"{savepath}/{word}.jpg")

        # Convert images to tensors
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np)
        tensors.append(img_tensor)
    
    return tensors

# NOTE: This function needs to be checked, missing image transformations
def get_image_train_data(self):
    grapheme_tensors = self.text_to_grapheme(self.words, self.savepath)
    grapheme_dataset = TensorDataset(*grapheme_tensors)
    grapheme_dataloader = DataLoader(grapheme_dataset, batch_size=self.batch_size, 
                                        shuffle=True, drop_last=True)

    return grapheme_dataloader