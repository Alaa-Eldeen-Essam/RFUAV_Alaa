# A script to do the inference on the spectrogram image or binary raw frequency data pack using trained classify model or two-stage detector model.

from utils.benchmark import Classify_Model
from utils.TwoStagesDetector import TwoStagesDetector

def main():

     # doing a inference on spectrogram image or binary raw frequency data pack using the trained classify model
    source = r'D:\Behoos_AI\AI Projects\DF\detect_classify\RFUAV_Alaa\example\example.png'
    test = Classify_Model(cfg=r'D:\Behoos_AI\AI Projects\DF\detect_classify\RFUAV_Alaa\configs\exp2.10_vit_l_16_hot.yaml',
                          weight_path=r'D:\Behoos_AI\AI Projects\DF\detect_classify\Data\RFUAV\weight\exp2\hot\vit_l_16.pth')
    test.inference(source=source, save_path='./res/')  # for inference test

    # doing a two-stage detector inference on the binary raw frequency data pack using the trained detector and classify model
    cfg_path = '../example/two_stage/sample.json'
    TwoStagesDetector(cfg=cfg_path)


if __name__ == '__main__':
    main()