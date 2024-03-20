from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
import nltk

# 确保已经下载了NLTK的分词器数据
nltk.download('punkt')

# 初始化词干提取器
stemmer = LancasterStemmer()

# 示例句子
sentence = "portugal"

# portugal
# portuguese

# 分词
words = word_tokenize(sentence)

# 对每个单词提取词干
stemmed_words = [stemmer.stem(word) for word in words]

print("Original words:", words)
print("Stemmed words:", stemmed_words)
