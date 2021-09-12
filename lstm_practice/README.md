# このフォルダ内のプログラムについて

## 1.anacondaの仮想環境をコピー
以下のコマンドをターミナル上で打つとanacondaの環境ファイルとこの記事で紹介するプログラムを含むリポジトリをダウンロードできます．

```bash
git clone https://github.com/sloth-hobby/PyTorch_practice.git
```
ターミナル上で`PyTorch_practice/lstm_practice/envs/`のディレクトリ下に移動し，anacondaの仮想環境に入っている状態で以下のコマンドを打つと仮想環境をコピーできます．環境名のところはどんな名前でも良いです．

```bash
conda env create -n 環境名 -f predict_simple_formula_env.yml
```

## 2.プログラムの実行
1.で作成した仮想環境に入っている状態で，`PyTorch_practice/lstm_practice/`のディレクトリ下に移動し，以下のコマンドのように打つと実行できます．
```bash
python “実行したいプログラムファイル”
```
## 3.プログラムの説明
プログラムの詳しい説明は以下のqiitaの記事に書きました．

以下の記事は，`PyTorch_practice/lstm_practice/`のディレクトリにある`predict_simple_formula_train.py`というプログラムを説明する記事です．   
- [[PyTorch1.9.0]LSTMを使って時系列(単純な数式)予測してみた](https://qiita.com/sloth-hobby/items/93982c79a70b452b2e0a)

以下の記事は，`PyTorch_practice/lstm_practice/`のディレクトリにある`predict_simple_formula_train_m_to_m.py`というプログラムを説明する記事です．   
- [[PyTorch 1.9.0] LSTMを使っていくつ先の未来まで精度良く予測できるのか検証してみた](https://qiita.com/sloth-hobby/items/8d4c7c1185d660921b7a)
