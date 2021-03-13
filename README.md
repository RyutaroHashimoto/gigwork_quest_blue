# gigwork_quest_blue
team repository for gigwork quest

## コードの説明

- preprocessing.ipynb
  - データ拡張とデータセットの読み込み
- EfficientNetB7.ipynb
  - EfficientNetを素の状態で訓練したもの
  - 参考サイト
    - https://www.kaggle.com/micajoumathematics/fine-tuning-efficientnetb0-on-cifar-100
- EfficientNetB7_apply_ImageDataGenerator.ipynb
  - データを拡張し、モデルを訓練したもの
- RandAugument.ipynb
  - RandAugumentを用いてモデルを訓練したもの
  - 参考サイト
    - https://qiita.com/rabbitcaptain/items/a15591ca49dc428223ca
- RandAugument_GridSearch.ipynb
  - RandAugumentをグリッドサーチ（グリッドサーチする所まで含めてRandAugumentだと思いますので頭痛が痛いみたいになっています。。）
- save_npy.ipynb
  - 訓練データをnpyで保存
  - Google Colab ではデータロード時にフリーズしてしまう可能性が高いため作成