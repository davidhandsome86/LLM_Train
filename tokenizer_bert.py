
# bert的分词模型用的是WordPiece

from tokenizers import Tokenizer,processors
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
files =  ["./sanguo.txt"] #你的训练数据 your training_data

tokenizer.normalizer = BertNormalizer(lowercase=True)

tokenizer.pre_tokenizer = BertPreTokenizer()

special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = WordPieceTrainer(vocab_size=50000, show_progress=True, special_tokens=special_tokens)
tokenizer.train(files, trainer)

cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)
tokenizer.decoder = WordPieceDecoder(prefix="##")

tokenizer.save("tokenizer_text.json")

# 如果要在Transformers中使用这个分词器，我们需要将它包装在一个PreTrainedTokenizerFast中。在这我们使用特定的标记器类BertTokenizerFast
from transformers import BertTokenizerFast

wrapped_tokenizer = BertTokenizerFast(tokenizer_object = tokenizer)
wrapped_tokenizer.save_pretrained("./bert")


