import json
from pathlib import Path
import tempfile
import unittest
try:
    import torch
    from PIL import Image
    from training.freeview_data import FreeviewDataset
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'model environment required')
class FreeviewTests(unittest.TestCase):
    def test_canvas_window_target_and_padding(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);Image.new('RGB',(1680,1050)).save(root/'1.jpg')
            source=root/'fix.json';source.write_text(json.dumps([dict(name='1.jpg',split='train',length=9,X=[840.]*9,Y=[525.]*9,T=[100.]*9)]))
            ds=FreeviewDataset(source,root,'train');_,seq,mask,target=ds[0]
            self.assertEqual(mask.tolist(),[False,True,True,True,True,True])
            self.assertEqual(seq[0].item(),40*128+64+1);self.assertEqual(target.item(),40*128+64)
            self.assertFalse(ds[7][2].any());self.assertEqual(len(ds),8)
    def test_reject_unpadded_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);Image.new('RGB',(100,100)).save(root/'1.jpg')
            source=root/'fix.json';source.write_text(json.dumps([dict(name='1.jpg',split='train',length=2,X=[1,2],Y=[1,2],T=[10,10])]))
            ds=FreeviewDataset(source,root,'train')
            with self.assertRaisesRegex(ValueError,'canvas'):ds[0]
