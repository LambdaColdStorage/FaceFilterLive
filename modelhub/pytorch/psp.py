from pathlib import Path
import torch
import numpy as np
import sys
sys.path.append("psp")
import editor


class PspEditor():
    def __init__(self) -> None:
        super().__init__()
        checkpoint_path = Path(__file__).parent / "psp_ffhq_encode.pt"
        encoder, decoder, latent_avg = editor.load_model(checkpoint_path)

        manipulator = editor.manipulate_model(decoder)
        manipulator.edits = {editor.idx_dict[v[0]]: {v[1]: 0} for k, v in editor.edits.items()}

        age_path = Path(__file__).parent / "age.pt"
        self.age_edit = torch.load(age_path)
        
        self.model = {
            "encoder": encoder,
            "decoder": decoder,
            "latent_avg": latent_avg,
            "manipulator": manipulator,
        }

    def run(self, inp, edits):

        # age is a different type of edit to the others
        age_scale = edits.pop("age")

        manipulator = self.model["manipulator"]
        for k, v in editor.edits.items():
            layer_index, channel_index, sense = v
            conv_name = editor.idx_dict[layer_index]
            manipulator.edits[conv_name][channel_index] = edits.get(k, 0)*sense

        inp = 2*inp[...,::-1].astype(np.float32).transpose(2,0,1)/255 - 1
        inp = torch.tensor(inp.copy())
        output = editor.run(
            self.model["encoder"],
            self.model["decoder"],
            self.model["latent_avg"],
            inp,
            edit=age_scale*self.age_edit.to("cuda"),
            output_pil=False,
            input_is_pil=False,
            )
        output = np.ascontiguousarray(output[...,::-1]).astype(np.uint8)
        return output