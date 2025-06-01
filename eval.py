import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import nltk
from collections import Counter
from textstat import textstat
from bert_score import score as bert_score
from evaluate import load

nltk.download('punkt')

class DentalDatasetEvaluator:
    def __init__(self, dataset_name="NV9523/DentalGPT_SFT", sample_size=None):
        """
        Khởi tạo bộ đánh giá dataset
        Args:
            dataset_name: Tên dataset trên HuggingFace Hub
            sample_size: Số lượng mẫu cần đánh giá (None để đánh giá toàn bộ)
        """
        self.dataset = self._load_dataset(dataset_name, sample_size)
        self.metrics = {
            'bleu': load("bleu"),
            'rouge': load("rouge"),
            'readability': textstat
        }
        self.results = {}

    def _load_dataset(self, dataset_name, sample_size):
        """Tải dataset với kích thước mẫu tuỳ chọn"""
        split = f"train[:{sample_size}]" if sample_size else "train"
        return load_dataset(dataset_name, split=split)

    def _calculate_text_metrics(self, text):
        """Tính các chỉ số về độ phức tạp văn bản"""
        return {
            'word_count': len(nltk.word_tokenize(text)),
            'char_count': len(text),
            'sentence_count': len(nltk.sent_tokenize(text)),
            'flesch_reading_ease': self.metrics['readability'].flesch_reading_ease(text),
            'lexical_diversity': len(set(nltk.word_tokenize(text))) / len(nltk.word_tokenize(text))
        }

    def _analyze_content_distribution(self):
        """Phân tích phân phối nội dung trong dataset"""
        topics = [example['Chủ đề'] for example in self.dataset]
        return dict(Counter(topics))

    def _evaluate_qa_consistency(self):
        """Đánh giá tính nhất quán giữa câu hỏi và câu trả lời"""
        questions = [example['Câu hỏi'] for example in self.dataset]
        answers = [example['Câu trả lời'] for example in self.dataset]
        
        # Tính BERTScore cho QA consistency
        P, R, F1 = bert_score(answers, questions, lang="vi", verbose=False)
        return {
            'bert_score_f1': F1.mean().item(),
            'qa_length_ratio': np.mean([len(a)/len(q) for q, a in zip(questions, answers)])
        }

    def _evaluate_cot_components(self):
        """Đánh giá các thành phần Chain-of-Thought"""
        cot_metrics = {
            'goal_length': [],
            'reasoning_length': [],
            'justification_length': []
        }
        
        for example in self.dataset:
            cot_metrics['goal_length'].append(len(example['CoT_Goal'].split()))
            cot_metrics['reasoning_length'].append(len(example['CoT_Reasoning'].split()))
            cot_metrics['justification_length'].append(len(example['CoT_Justification'].split()))
        
        return {k: np.mean(v) for k, v in cot_metrics.items()}

    def generate_report(self):
        """Tạo báo cáo đánh giá toàn diện"""
        print("🔍 Bắt đầu đánh giá bộ dữ liệu DentalGPT...")
        
        # 1. Phân tích cơ bản
        self.results['basic_stats'] = {
            'dataset_size': len(self.dataset),
            'columns': list(self.dataset.features.keys())
        }

        # 2. Phân tích phân phối nội dung
        self.results['content_distribution'] = self._analyze_content_distribution()

        # 3. Đánh giá chất lượng văn bản
        print("\n📝 Đang phân tích chất lượng văn bản...")
        sample_text = self.dataset[0]['Câu trả lời']
        self.results['text_quality'] = self._calculate_text_metrics(sample_text)

        # 4. Đánh giá QA consistency
        print("\n🔗 Đang đánh giá tính nhất quán QA...")
        self.results['qa_consistency'] = self._evaluate_qa_consistency()

        # 5. Phân tích thành phần CoT
        print("\n🧠 Đang phân tích Chain-of-Thought...")
        self.results['cot_analysis'] = self._evaluate_cot_components()

        # Xuất báo cáo
        self._print_report()

        return self.results

    def _print_report(self):
        """In báo cáo định dạng đẹp"""
        print("\n📊 BÁO CÁO ĐÁNH GIÁ BỘ DỮ LIỆU")
        print("="*60)
        
        # 1. Thông tin cơ bản
        print(f"\n📌 Tổng số mẫu: {self.results['basic_stats']['dataset_size']}")
        print(f"📌 Các trường dữ liệu: {', '.join(self.results['basic_stats']['columns'])}")
        
        # 2. Phân phối chủ đề
        print("\n🌐 Phân phối chủ đề:")
        for topic, count in self.results['content_distribution'].items():
            print(f"  - {topic}: {count} mẫu ({count/len(self.dataset)*100:.1f}%)")
        
        # 3. Chất lượng văn bản
        print("\n📖 Chất lượng văn bản (ví dụ mẫu):")
        print(f"  - Số từ: {self.results['text_quality']['word_count']}")
        print(f"  - Độ dài câu: {self.results['text_quality']['sentence_count']}")
        print(f"  - Độ phức tạp (Flesch): {self.results['text_quality']['flesch_reading_ease']:.1f}")
        print(f"  - Đa dạng từ vựng: {self.results['text_quality']['lexical_diversity']:.2f}")
        
        # 4. QA Consistency
        print("\n🔗 Chỉ số nhất quán QA:")
        print(f"  - BERTScore F1: {self.results['qa_consistency']['bert_score_f1']:.3f}")
        print(f"  - Tỷ lệ độ dài trả lời/câu hỏi: {self.results['qa_consistency']['qa_length_ratio']:.2f}")
        
        # 5. Phân tích CoT
        print("\n🧠 Phân tích Chain-of-Thought:")
        print(f"  - Độ dài trung bình mục tiêu: {self.results['cot_analysis']['goal_length']:.1f} từ")
        print(f"  - Độ dài trung bình lập luận: {self.results['cot_analysis']['reasoning_length']:.1f} từ")
        print(f"  - Độ dài trung bình giải thích: {self.results['cot_analysis']['justification_length']:.1f} từ")

if __name__ == "__main__":
    # Khởi tạo và chạy đánh giá
    evaluator = DentalDatasetEvaluator(sample_size=100)
    report = evaluator.generate_report()