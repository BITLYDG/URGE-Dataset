from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import ast
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from prompt_template import getInstruction
import argparse


@dataclass
class ModelConfig:
    """Configuration for model initialization"""
    model_path: str
    tensor_parallel_size: int = 4
    enforce_eager: bool = True
    gpu_memory_utilization: float = 0.98


@dataclass
class GenerationConfig:
    """Configuration for text generation parameters"""
    temperature: float = 0.0
    top_p: float = 0.8
    max_tokens: int = 1000000
    batch_size: int = 4


class ModelHandler:
    """Handles model initialization and text generation"""
    
    def __init__(self, config: ModelConfig):
        self.tokenizer, self.llm = self._initialize_model(config)
        
    @staticmethod
    def _initialize_model(config: ModelConfig) -> Tuple[AutoTokenizer, LLM]:
        """Initialize model and tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        llm = LLM(
            model=config.model_path,
            enforce_eager=config.enforce_eager,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization
        )
        return tokenizer, llm

    def generate(self, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
        """Generate responses for a batch of prompts"""
        formatted_inputs = self._format_inputs(prompts)
        outputs = self.llm.generate(formatted_inputs, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def _format_inputs(self, prompts: List[str]) -> List[str]:
        """Format prompts with system message"""
        system_message = "You are an AI response evaluation expert skilled in assessing AI-generated responses to users."
        return [
            self.tokenizer.apply_chat_template([
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ], tokenize=False, add_generation_prompt=True)
            for prompt in prompts
        ]


class DataProcessor:
    """Handles data loading, processing, and result validation"""
    
    def __init__(self, prompt_column: str = 'prompt', target_column: str = 'evaluation'):
        self.prompt_column = prompt_column
        self.target_column = target_column

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and initialize evaluation dataframe"""
        df = pd.read_csv(file_path)
        if self.target_column not in df.columns:
            df[self.target_column] = ""
        return df.fillna('')

    def generate_prompts(self, df: pd.DataFrame, template_type: str) -> List[str]:
        """Generate prompts based on specified template"""
        if template_type == 'URS':
            return [
                getInstruction(dialog, template_type, task_type)
                for dialog, task_type in zip(
                    df[self.prompt_column], df['task_type']
                )
            ]
        return [
            getInstruction(dialog, template_type)
            for dialog in df[self.prompt_column]
        ]

    @staticmethod
    def validate_output(text: str, expected_length: int, template_type: str) -> bool:
        """Validate output format based on template type"""
        if template_type == 'G-eval':
            return DataValidator.validate_list_format(text, expected_length)
        return DataValidator.validate_json_format(text, expected_length)

    def clean_invalid_results(self, file_path: str) -> int:
        """Clean and revalidate results in output file"""
        df = pd.read_csv(file_path)
        invalid_count = 0
        
        for idx in tqdm(df.index, desc="Validating results"):
            if not self._validate_row(df.loc[idx]):
                df.at[idx, self.target_column] = ""
                invalid_count += 1
                
        df.to_csv(file_path, encoding='utf-8-sig', index=False)
        return invalid_count

    def _validate_row(self, row: pd.Series) -> bool:
        """Validate individual row result"""
        try:
            turn_count = len(ast.literal_eval(row['sat_turn'])) + 1
            result = ast.literal_eval(row[self.target_column])
            
            if isinstance(result, list):
                return len(result) == turn_count
            if isinstance(result, dict):
                return all(len(v) == 6 for v in result.values())
            return False
        except (SyntaxError, ValueError):
            return False


class DataValidator:
    """Static methods for output validation"""
    
    @staticmethod
    def validate_list_format(text: str, expected_length: int) -> bool:
        """Validate list format output"""
        result = DataValidator._extract_list(text)
        return isinstance(result, list) and len(result) == expected_length

    @staticmethod
    def validate_json_format(text: str, expected_length: int) -> bool:
        """Validate JSON format output"""
        result = DataValidator._extract_json(text)
        return (
            isinstance(result, dict) and 
            len(result) == expected_length and 
            all(len(v) == 6 for v in result.values())
        )

    @staticmethod
    def _extract_list(text: str) -> Optional[List]:
        """Extract outermost list from text"""
        try:
            return ast.literal_eval(text[text.index('['):text.rindex(']')+1])
        except (ValueError, SyntaxError):
            return None

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """Extract outermost JSON object from text"""
        clean_text = (
            text.replace("\n", "")
            .replace("‘", "'").replace("’", "'")
            .replace("```", "").strip('"').strip("'").strip(",")
        )
        try:
            return ast.literal_eval(clean_text[clean_text.index('{'):clean_text.rindex('}')+1])
        except (ValueError, SyntaxError):
            return None


class EvaluationPipeline:
    """Main execution pipeline for evaluation generation"""
    
    def __init__(self, model_config: ModelConfig, gen_config: GenerationConfig):
        self.model = ModelHandler(model_config)
        self.sampling_params = SamplingParams(
            temperature=gen_config.temperature,
            top_p=gen_config.top_p,
            max_tokens=gen_config.max_tokens
        )
        self.batch_size = gen_config.batch_size

    def execute(
        self,
        input_file: str,
        output_file: str,
        template_type: str,
        processor: DataProcessor
    ) -> None:
        """Execute full evaluation pipeline"""
        df = processor.load_data(input_file)
        prompts = processor.generate_prompts(df, template_type)
        
        for i in tqdm(range(0, len(df), self.batch_size), desc="Processing batches"):
            batch = self._process_batch(df.iloc[i:i+self.batch_size], prompts, template_type, processor)
            df.update(batch)
            
        processor.save_results(df, output_file)

    def _process_batch(
        self,
        batch: pd.DataFrame,
        prompts: List[str],
        template_type: str,
        processor: DataProcessor
    ) -> pd.DataFrame:
        """Process individual batch"""
        incomplete = batch[batch[processor.target_column] == ""]
        if incomplete.empty:
            return pd.DataFrame()

        batch_prompts = [prompts[idx] for idx in incomplete.index]
        responses = self.model.generate(batch_prompts, self.sampling_params)
        
        for idx, response in zip(incomplete.index, responses):
            turn_length = len(ast.literal_eval(batch.loc[idx, 'sat_turn'])) + 1
            if processor.validate_output(response, turn_length, template_type):
                batch.at[idx, processor.target_column] = response
        return batch


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="LLM Evaluation Pipeline")
    
    # Model configuration
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to pretrained model')
    parser.add_argument('--tensor_parallel_size', type=int, default=4,
                      help='Tensor parallel size for model distribution')
    parser.add_argument('--enforce_eager', action='store_true',
                      help='Enable eager mode execution')
    
    # Data configuration
    parser.add_argument('--input_file', type=str, required=True,
                      help='Input CSV file path')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Output CSV file path')
    parser.add_argument('--prompt_column', type=str, default='prompt',
                      help='Column name containing prompt data')
    
    # Generation parameters
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Inference batch size')
    parser.add_argument('--max_tokens', type=int, default=1000000,
                      help='Maximum tokens per response')
    parser.add_argument('--temperature', type=float, default=0.0,
                      help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.8,
                      help='Top-p sampling value')
    
    # Template configuration
    parser.add_argument('--template_type', type=str, required=True,
                      choices=['G-eval', 'URS'],
                      help='Prompt template type')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    model_config = ModelConfig(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager
    )
    
    gen_config = GenerationConfig(
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    processor = DataProcessor(prompt_column=args.prompt_column)
    pipeline = EvaluationPipeline(model_config, gen_config)
    
    pipeline.execute(
        input_file=args.input_file,
        output_file=args.output_file,
        template_type=args.template_type,
        processor=processor
    )

