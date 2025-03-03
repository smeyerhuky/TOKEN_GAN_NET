#!/bin/bash

# GAN Tokenizer System - GAN Network Setup Script
# This script creates the core Python modules for the GAN network

set -e  # Exit on error

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

# Progress function
progress() {
  local width=50
  local percent=$1
  local chars=$((percent * width / 100))
  local spaces=$((width - chars))
  printf "${BLUE}Progress: [${GREEN}"
  printf "%${chars}s" | tr ' ' '█'
  printf "${BLUE}%${spaces}s] ${percent}%%${NC}\n" | tr ' ' '.'
}

echo -e "${BOLD}${BLUE}=== GAN Network Setup ===${NC}"
progress 5

# Create schema_interpreter.py
echo -e "${BLUE}Creating schema_interpreter.py...${NC}"
cat > schema_interpreter.py << 'EOF'
"""
GAN Schema Interpreter
---------------------
This module interprets GAN schema definitions from YAML files and builds
corresponding TensorFlow models based on the specifications.
"""

import os
import yaml
import json
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import numpy as np

logger = logging.getLogger('gan_tokenizer.schema_interpreter')

class GANSchemaValidator:
    """Validates GAN schema format and parameters."""
    
    @staticmethod
    def validate_schema(schema):
        """Validate that the schema has all required sections and valid parameters."""
        required_sections = ['generator', 'discriminator', 'training', 'encoding_operations']
        
        # Check if schema has the main gan_schema section
        if 'gan_schema' not in schema:
            raise ValueError("Schema must have a 'gan_schema' section")
        
        gan_schema = schema['gan_schema']
        
        # Check if all required sections are present
        for section in required_sections:
            if section not in gan_schema:
                raise ValueError(f"Schema is missing required section: '{section}'")
        
        # Validate generator section
        generator = gan_schema['generator']
        if 'latent_dim' not in generator:
            raise ValueError("Generator schema must specify 'latent_dim'")
        if 'layers' not in generator or not isinstance(generator['layers'], list):
            raise ValueError("Generator schema must contain a 'layers' list")
        
        # Validate discriminator section
        discriminator = gan_schema['discriminator']
        if 'layers' not in discriminator or not isinstance(discriminator['layers'], list):
            raise ValueError("Discriminator schema must contain a 'layers' list")
        
        # Validate training section
        training = gan_schema['training']
        required_train_params = ['optimizer', 'learning_rate', 'batch_size', 'epochs']
        for param in required_train_params:
            if param not in training:
                raise ValueError(f"Training schema is missing required parameter: '{param}'")
        
        # Validate encoding operations
        encoding_ops = gan_schema['encoding_operations']
        if not isinstance(encoding_ops, list) or len(encoding_ops) == 0:
            raise ValueError("Schema must define at least one encoding operation")
        
        for op in encoding_ops:
            if 'name' not in op:
                raise ValueError("Each encoding operation must have a 'name'")
        
        logger.info("Schema validation successful")
        return True


class GANSchemaInterpreter:
    """Interprets GAN schemas and builds corresponding models."""
    
    def __init__(self, schema_path):
        """Initialize with a schema file path."""
        self.schema_path = schema_path
        self.schema = None
        self.load_schema()
        
    def load_schema(self):
        """Load and validate the schema from file."""
        try:
            with open(self.schema_path, 'r') as f:
                self.schema = yaml.safe_load(f)
            
            # Validate schema
            GANSchemaValidator.validate_schema(self.schema)
            
            logger.info(f"Successfully loaded schema from {self.schema_path}")
        except Exception as e:
            logger.error(f"Error loading schema: {e}")
            raise
    
    def build_generator(self):
        """Build generator model from schema."""
        gen_schema = self.schema['gan_schema']['generator']
        latent_dim = gen_schema['latent_dim']
        
        model = keras.Sequential(name="Generator")
        
        # Add layers according to schema
        first_layer = True
        for layer_spec in gen_schema['layers']:
            if first_layer:
                # First layer needs input shape
                if layer_spec['type'] == 'dense':
                    model.add(layers.Dense(
                        layer_spec['units'], 
                        input_shape=(latent_dim,),
                        activation=self._get_activation(layer_spec.get('activation'))
                    ))
                first_layer = False
            else:
                # Add subsequent layers
                if layer_spec['type'] == 'dense':
                    model.add(layers.Dense(
                        layer_spec['units'],
                        activation=self._get_activation(layer_spec.get('activation'))
                    ))
                elif layer_spec['type'] == 'batch_norm':
                    model.add(layers.BatchNormalization())
                elif layer_spec['type'] == 'dropout':
                    model.add(layers.Dropout(layer_spec['rate']))
                elif layer_spec['type'] == 'reshape':
                    model.add(layers.Reshape(layer_spec['target_shape']))
        
        logger.info("Generator model built successfully")
        return model
    
    def build_discriminator(self, input_shape):
        """Build discriminator model from schema."""
        disc_schema = self.schema['gan_schema']['discriminator']
        
        model = keras.Sequential(name="Discriminator")
        
        # Add layers according to schema
        first_layer = True
        for layer_spec in disc_schema['layers']:
            if first_layer:
                # First layer needs input shape
                if layer_spec['type'] == 'dense':
                    model.add(layers.Dense(
                        layer_spec['units'], 
                        input_shape=input_shape,
                        activation=self._get_activation(layer_spec.get('activation'))
                    ))
                elif layer_spec['type'] == 'flatten':
                    model.add(layers.Flatten(input_shape=input_shape))
                first_layer = False
            else:
                # Add subsequent layers
                if layer_spec['type'] == 'dense':
                    model.add(layers.Dense(
                        layer_spec['units'],
                        activation=self._get_activation(layer_spec.get('activation'))
                    ))
                elif layer_spec['type'] == 'batch_norm':
                    model.add(layers.BatchNormalization())
                elif layer_spec['type'] == 'dropout':
                    model.add(layers.Dropout(layer_spec['rate']))
        
        logger.info("Discriminator model built successfully")
        return model
    
    def _get_activation(self, activation_name):
        """Map activation names to functions."""
        if not activation_name:
            return None
            
        activations = {
            'relu': 'relu',
            'leaky_relu': layers.LeakyReLU(alpha=0.2),
            'sigmoid': 'sigmoid',
            'tanh': 'tanh',
            'softmax': 'softmax',
            'linear': None
        }
        return activations.get(activation_name.lower(), None)
    
    def get_optimizer(self):
        """Get optimizer from schema."""
        training_schema = self.schema['gan_schema']['training']
        optimizer_name = training_schema['optimizer'].lower()
        lr = training_schema['learning_rate']
        
        optimizers = {
            'adam': optimizers.Adam(learning_rate=lr, beta_1=0.5),
            'sgd': optimizers.SGD(learning_rate=lr),
            'rmsprop': optimizers.RMSprop(learning_rate=lr),
            'adagrad': optimizers.Adagrad(learning_rate=lr)
        }
        
        optimizer = optimizers.get(optimizer_name)
        if optimizer is None:
            logger.warning(f"Unknown optimizer '{optimizer_name}', defaulting to Adam")
            optimizer = optimizers.Adam(learning_rate=lr)
        
        return optimizer
    
    def get_training_params(self):
        """Get training parameters from schema."""
        training_schema = self.schema['gan_schema']['training']
        return {
            'batch_size': training_schema['batch_size'],
            'epochs': training_schema['epochs'],
            'learning_rate': training_schema['learning_rate']
        }
    
    def get_encoding_operations(self):
        """Get encoding operations from schema."""
        return self.schema['gan_schema']['encoding_operations']
    
    def interpret_encoding_operations(self, raw_params, vocab_size, max_length):
        """
        Interpret raw parameters from generator to create encoding operations.
        
        Args:
            raw_params: Raw output from generator model
            vocab_size: Size of vocabulary
            max_length: Maximum length of sentences
            
        Returns:
            List of encoding operation dictionaries
        """
        encoding_ops = self.get_encoding_operations()
        num_ops = len(encoding_ops)
        
        # Reshape raw_params if needed
        if isinstance(raw_params, np.ndarray) and raw_params.ndim == 1:
            # Let's assume we need 4 parameters per operation
            params_per_op = 4
            raw_params = raw_params.reshape(-1, params_per_op)
        
        # Limit to number of operations in schema
        if len(raw_params) > num_ops:
            raw_params = raw_params[:num_ops]
        
        operations = []
        for i, op_params in enumerate(raw_params):
            op_type = encoding_ops[i % num_ops]['name']
            
            # Create operation based on type
            if op_type == 'shift':
                # Shift values by a constant
                param_range = encoding_ops[i % num_ops].get('param_range', [0, vocab_size])
                shift_value = int(np.interp(op_params[0], [0, 1], param_range))
                operations.append({
                    'type': 'shift',
                    'value': shift_value
                })
            elif op_type == 'multiply':
                # Multiply by a constant
                param_range = encoding_ops[i % num_ops].get('param_range', [1, 10])
                mult_value = int(np.interp(op_params[0], [0, 1], param_range))
                operations.append({
                    'type': 'multiply',
                    'value': max(1, mult_value)  # At least 1
                })
            elif op_type == 'swap':
                # Swap positions
                pos1 = int(op_params[0] * max_length)
                pos2 = int(op_params[1] * max_length)
                operations.append({
                    'type': 'swap',
                    'positions': [pos1, pos2]
                })
            elif op_type == 'transform':
                # Apply a mathematical transformation
                degree = encoding_ops[i % num_ops].get('degree', 2)
                a = int(op_params[0] * 10)  # Coefficient a
                b = int(op_params[1] * vocab_size)  # Coefficient b
                operations.append({
                    'type': 'transform',
                    'degree': degree,
                    'a': a,
                    'b': b
                })
        
        return operations
    
    def save_model_architecture(self, model, file_path):
        """Save model architecture to JSON file."""
        try:
            # Get model config
            model_config = model.get_config()
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            
            logger.info(f"Model architecture saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving model architecture: {e}")
EOF

progress 15

# Create progress_visualization.py
echo -e "${BLUE}Creating progress_visualization.py...${NC}"
cat > progress_visualization.py << 'EOF'
"""
Progress Visualization Module for GAN Tokenization System
---------------------------------------------------------
This module provides real-time progress visualization for the GAN training process,
including progress bars, statistics displays, and resource monitoring.
"""

import sys
import time
import os
import datetime
from typing import Dict, List, Optional, Union, Any
import numpy as np
import GPUtil
import psutil
from tqdm import tqdm
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init()

class ProgressVisualizer:
    """
    Handles visualization of training progress in the terminal.
    
    This class provides methods for showing:
    - Progress bars for epochs and batches
    - Loss and metric statistics
    - GPU/CPU resource usage
    - Time estimates
    - LLM evaluation feedback
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the progress visualizer.
        
        Args:
            config: Configuration dictionary with logging/visualization settings
        """
        self.config = config
        self.logging_config = config.get('logging', {})
        self.verbosity = self.logging_config.get('verbosity', 1)  # 0=minimal, 1=normal, 2=detailed
        self.show_progress_bar = self.logging_config.get('show_progress_bar', True)
        self.update_interval = self.logging_config.get('update_interval', 1)
        self.show_resource_usage = self.logging_config.get('show_resource_usage', True)
        self.show_examples = self.logging_config.get('show_examples', True)
        
        # Training stats
        self.epoch_start_time = None
        self.training_start_time = None
        self.current_epoch = 0
        self.total_epochs = 0
        self.epoch_progress_bar = None
        self.batch_progress_bar = None
        
        # Resource monitoring
        self.last_resource_check = 0
        self.last_resource_values = {}
    
    def start_training(self, total_epochs: int) -> None:
        """
        Start training visualization.
        
        Args:
            total_epochs: Total number of epochs for training
        """
        self.training_start_time = time.time()
        self.total_epochs = total_epochs
        
        self._print_header()
    
    def start_epoch(self, epoch: int, total_batches: int) -> None:
        """
        Start visualization for a new epoch.
        
        Args:
            epoch: Current epoch number
            total_batches: Total number of batches in this epoch
        """
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        
        # Create epoch progress bar if needed
        if self.show_progress_bar:
            # Close previous progress bar if it exists
            if self.batch_progress_bar is not None:
                self.batch_progress_bar.close()
            
            # Create new progress bar for this epoch
            self.batch_progress_bar = tqdm(
                total=total_batches,
                desc=f"Epoch {epoch}/{self.total_epochs}",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                file=sys.stdout,
                dynamic_ncols=True
            )
    
    def update_batch(self, 
                    batch: int, 
                    losses: Dict[str, float], 
                    metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Update visualization for a batch.
        
        Args:
            batch: Current batch number
            losses: Dictionary of loss values
            metrics: Dictionary of metric values (optional)
        """
        # Update progress bar if enabled
        if self.show_progress_bar and self.batch_progress_bar is not None:
            self.batch_progress_bar.update(1)
            
            # Update progress bar description with losses
            desc = f"Epoch {self.current_epoch}/{self.total_epochs}"
            if self.verbosity > 0:
                loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
                desc += f" - {loss_str}"
                
                if metrics and self.verbosity > 1:
                    metric_str = ", ".join([f"{k}: {v:.2f}" for k, v in metrics.items()])
                    desc += f" - {metric_str}"
            
            self.batch_progress_bar.set_description(desc)
        
        # Show resource usage if enabled and it's time to update
        if (self.show_resource_usage and 
            time.time() - self.last_resource_check > self.update_interval):
            self._update_resource_usage()
    
    def end_epoch(self, 
                 losses: Dict[str, float], 
                 metrics: Optional[Dict[str, float]] = None,
                 llm_feedback: Optional[Dict[str, Any]] = None) -> None:
        """
        End visualization for an epoch.
        
        Args:
            losses: Dictionary of loss values for the epoch
            metrics: Dictionary of metric values for the epoch (optional)
            llm_feedback: Dictionary with LLM evaluation results (optional)
        """
        # Close batch progress bar if it exists
        if self.show_progress_bar and self.batch_progress_bar is not None:
            self.batch_progress_bar.close()
            self.batch_progress_bar = None
        
        # Calculate time taken for this epoch
        epoch_time = time.time() - self.epoch_start_time
        
        # Print epoch summary
        self._print_epoch_summary(losses, metrics, epoch_time, llm_feedback)
    
    def end_training(self, 
                    final_losses: Dict[str, float], 
                    final_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        End training visualization.
        
        Args:
            final_losses: Dictionary of final loss values
            final_metrics: Dictionary of final metric values (optional)
        """
        # Calculate total training time
        total_time = time.time() - self.training_start_time
        
        # Print training summary
        self._print_training_summary(final_losses, final_metrics, total_time)
    
    def display_llm_evaluation(self, 
                              original_text: str, 
                              encoded_text: str, 
                              schema: Dict, 
                              result: Dict) -> None:
        """
        Display LLM evaluation results.
        
        Args:
            original_text: Original text that was encoded
            encoded_text: Encoded text representation
            schema: Encoding schema used
            result: Dictionary with LLM evaluation results
        """
        if self.verbosity < 1:
            return
            
        score = result.get('score', 0)
        feedback = result.get('feedback', '')
        
        print(f"\n{Fore.CYAN}LLM Evaluation Results:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Original: {original_text[:50]}...{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Encoded : {str(encoded_text)[:50]}...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Score   : {score:.2f}/1.0{Style.RESET_ALL}")
        
        if self.verbosity > 1:
            # Truncate feedback if it's too long
            if len(feedback) > 200:
                feedback = feedback[:200] + "..."
            print(f"{Fore.WHITE}Feedback: {feedback}{Style.RESET_ALL}")
            
            # Show schema summary if verbosity is high
            print(f"{Fore.WHITE}Schema  : {len(schema)} operations{Style.RESET_ALL}")
            if self.verbosity > 2:
                op_types = {}
                for op in schema:
                    op_type = op.get('type', 'unknown')
                    op_types[op_type] = op_types.get(op_type, 0) + 1
                
                print(f"{Fore.WHITE}Operations: {op_types}{Style.RESET_ALL}")
    
    def display_example(self, 
                       original: str, 
                       encoded: np.ndarray, 
                       decoded: str, 
                       score: float) -> None:
        """
        Display an example encoding.
        
        Args:
            original: Original text
            encoded: Encoded representation
            decoded: Decoded text (if available)
            score: Evaluation score
        """
        if not self.show_examples or self.verbosity < 1:
            return
            
        print(f"\n{Fore.GREEN}Example Encoding:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Original: {original[:50]}...{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Encoded : {str(encoded)[:50]}...{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Decoded : {decoded[:50]}...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Score   : {score:.2f}/1.0{Style.RESET_ALL}")
    
    def _print_header(self) -> None:
        """Print training header with information about the run."""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}=" * 80)
        print(f"GAN Tokenization System - Training Started")
        print(f"=" * 80{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Start time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total epochs: {self.total_epochs}")
        
        # Print hardware info if showing resource usage
        if self.show_resource_usage:
            self._print_hardware_info()
        
        print(f"{Fore.CYAN}{'-' * 80}{Style.RESET_ALL}\n")
    
    def _print_hardware_info(self) -> None:
        """Print information about available hardware."""
        # CPU info
        cpu_count = psutil.cpu_count(logical=True)
        cpu_physical = psutil.cpu_count(logical=False)
        memory = psutil.virtual_memory()
        
        print(f"{Fore.WHITE}CPU: {cpu_physical} physical cores, {cpu_count} logical cores")
        print(f"RAM: {memory.total / (1024**3):.1f} GB total, "
              f"{memory.available / (1024**3):.1f} GB available")
        
        # GPU info
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                for i, gpu in enumerate(gpus):
                    print(f"GPU {i}: {gpu.name}, {gpu.memoryTotal} MB VRAM, "
                          f"Driver: {gpu.driver}")
        except Exception:
            print("GPU information not available")
    
    def _update_resource_usage(self) -> None:
        """Update and display resource usage."""
        self.last_resource_check = time.time()
        
        # Get CPU and RAM usage
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Get GPU usage if available
        gpu_info = {}
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Assuming first GPU
                gpu_info = {
                    'load': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal
                }
        except Exception:
            pass
        
        # Store current values
        self.last_resource_values = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            **gpu_info
        }
        
        # Print resource usage if verbosity allows
        if self.verbosity > 0 and not self.show_progress_bar:
            self._print_resource_usage()
        elif self.show_progress_bar and self.batch_progress_bar is not None:
            # Add resource info to progress bar
            self.batch_progress_bar.set_postfix_str(self._get_resource_usage_str())
    
    def _print_resource_usage(self) -> None:
        """Print current resource usage."""
        values = self.last_resource_values
        
        print(f"{Fore.BLUE}Resource usage: "
              f"CPU: {values.get('cpu_percent', 'N/A'):.1f}%, "
              f"RAM: {values.get('memory_percent', 'N/A'):.1f}%", end='')
        
        # Add GPU info if available
        if 'load' in values:
            print(f", GPU: {values.get('load', 'N/A'):.1f}%, "
                  f"VRAM: {values.get('memory_used', 'N/A'):.0f}/"
                  f"{values.get('memory_total', 'N/A'):.0f} MB", end='')
        
        print(f"{Style.RESET_ALL}")
    
    def _get_resource_usage_str(self) -> str:
        """Get resource usage as a compact string for progress bar."""
        values = self.last_resource_values
        
        result = f"CPU: {values.get('cpu_percent', 'N/A'):.1f}%"
        
        # Add GPU info if available
        if 'load' in values:
            result += f", GPU: {values.get('memory_used', 'N/A'):.0f}/"
                      f"{values.get('memory_total', 'N/A'):.0f}MB"
        
        return result
    
    def _print_epoch_summary(self, 
                            losses: Dict[str, float], 
                            metrics: Optional[Dict[str, float]], 
                            epoch_time: float,
                            llm_feedback: Optional[Dict[str, Any]] = None) -> None:
        """Print summary for an epoch."""
        # Skip if verbosity is too low
        if self.verbosity < 1:
            return
            
        # Calculate progress and time estimates
        progress = (self.current_epoch + 1) / self.total_epochs
        elapsed_total = time.time() - self.training_start_time
        estimated_total = elapsed_total / progress if progress > 0 else 0
        remaining = max(0, estimated_total - elapsed_total)
        
        # Format losses and metrics
        losses_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
        metrics_str = ", ".join([f"{k}: {v:.2f}" for k, v in metrics.items()]) if metrics else ""
        
        # Print summary
        print(f"\n{Fore.GREEN}Epoch {self.current_epoch+1}/{self.total_epochs} completed "
              f"in {epoch_time:.2f}s{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Losses: {losses_str}{Style.RESET_ALL}")
        
        if metrics_str:
            print(f"{Fore.WHITE}Metrics: {metrics_str}{Style.RESET_ALL}")
        
        # Print time estimates
        print(f"{Fore.BLUE}Time elapsed: {self._format_time(elapsed_total)}, "
              f"remaining: {self._format_time(remaining)}{Style.RESET_ALL}")
        
        # Print resource usage
        if self.show_resource_usage:
            self._print_resource_usage()
        
        # Print LLM feedback if available and verbosity allows
        if llm_feedback and self.verbosity > 1:
            score = llm_feedback.get('score', 0)
            feedback = llm_feedback.get('feedback', '')
            
            # Truncate feedback if it's too long
            if len(feedback) > 100:
                feedback = feedback[:100] + "..."
                
            print(f"{Fore.CYAN}LLM Evaluation: Score {score:.2f}/1.0 - \"{feedback}\"{Style.RESET_ALL}")
        
        print(f"{Fore.CYAN}{'-' * 40}{Style.RESET_ALL}")
    
    def _print_training_summary(self, 
                               final_losses: Dict[str, float], 
                               final_metrics: Optional[Dict[str, float]], 
                               total_time: float) -> None:
        """Print summary for the entire training."""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}=" * 80)
        print(f"GAN Tokenization System - Training Completed")
        print(f"=" * 80{Style.RESET_ALL}")
        
        # Format final losses and metrics
        losses_str = ", ".join([f"{k}: {v:.4f}" for k, v in final_losses.items()])
        metrics_str = ", ".join([f"{k}: {v:.2f}" for k, v in final_metrics.items()]) if final_metrics else ""
        
        print(f"{Fore.WHITE}Total time: {self._format_time(total_time)}")
        print(f"Final losses: {losses_str}")
        
        if metrics_str:
            print(f"Final metrics: {metrics_str}")
        
        print(f"{Fore.GREEN}Training results saved to models/ and logs/ directories{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'-' * 80}{Style.RESET_ALL}\n")
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to hours:minutes:seconds format."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
EOF

progress 30

# Create corpus_manager.py
echo -e "${BLUE}Creating corpus_manager.py...${NC}"
cat > corpus_manager.py << 'EOF'
"""
Corpus Manager for GAN Tokenization System
----------------------------------------
This module manages text corpora using scikit-learn tools for preprocessing,
feature extraction, and enhancement for the GAN tokenization system.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import json

logger = logging.getLogger('gan_tokenizer.corpus_manager')

class TextPreprocessor:
    """Handles text preprocessing tasks."""
    
    def __init__(self, 
                 remove_stopwords: bool = True,
                 lemmatize: bool = True,
                 min_word_length: int = 2,
                 language: str = 'english'):
        """
        Initialize text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            min_word_length: Minimum word length to keep
            language: Language for stopwords and processing
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length
        self.language = language
        
        # Initialize NLTK components
        try:
            self.stop_words = set(stopwords.words(language)) if remove_stopwords else set()
            self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        except LookupError:
            logger.warning("NLTK resources not found. Downloading...")
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words(language)) if remove_stopwords else set()
            self.lemmatizer = WordNetLemmatizer() if lemmatize else None
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess a single text.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        words = word_tokenize(text)
        
        # Remove stopwords and short words
        if self.remove_stopwords:
            words = [w for w in words if w not in self.stop_words and len(w) >= self.min_word_length]
        
        # Lemmatize
        if self.lemmatize:
            words = [self.lemmatizer.lemmatize(w) for w in words]
        
        # Join back into a string
        return ' '.join(words)
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts]


class CorpusManager:
    """
    Manages text corpora for the GAN system, including loading, preprocessing,
    feature extraction, and enhancement using scikit-learn tools.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize corpus manager with configuration.
        
        Args:
            config: Configuration dictionary with data processing parameters
        """
        self.config = config
        self.data_config = config.get('data', {})
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor(
            remove_stopwords=self.data_config.get('remove_stopwords', True),
            lemmatize=self.data_config.get('lemmatize', True),
            min_word_length=self.data_config.get('min_word_length', 2)
        )
        
        # Initialize variables
        self.texts = []
        self.processed_texts = []
        self.encodings = None
        self.vectorizer = None
        self.svd = None
    
    def load_corpus(self, file_path: str) -> List[str]:
        """
        Load text corpus from file.
        
        Args:
            file_path: Path to the corpus file (one text per line)
            
        Returns:
            List of loaded texts
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Loaded {len(texts)} texts from {file_path}")
            self.texts = texts
            return texts
        except Exception as e:
            logger.error(f"Error loading corpus: {e}")
            return []
    
    def preprocess_corpus(self, texts: Optional[List[str]] = None) -> List[str]:
        """
        Preprocess the corpus.
        
        Args:
            texts: List of texts to preprocess, or None to use loaded texts
            
        Returns:
            List of preprocessed texts
        """
        if texts is None:
            texts = self.texts
        
        self.processed_texts = self.preprocessor.preprocess_texts(texts)
        logger.info(f"Preprocessed {len(self.processed_texts)} texts")
        return self.processed_texts
    
    def create_vectorizer(self) -> Union[CountVectorizer, TfidfVectorizer, HashingVectorizer]:
        """
        Create a vectorizer based on configuration.
        
        Returns:
            Configured vectorizer instance
        """
        vectorizer_type = self.data_config.get('vectorizer', 'tfidf').lower()
        max_features = self.data_config.get('max_features', 5000)
        
        if vectorizer_type == 'count':
            vectorizer = CountVectorizer(
                max_features=max_features,
                min_df=2,
                max_df=0.95
            )
        elif vectorizer_type == 'hash':
            vectorizer = HashingVectorizer(
                n_features=max_features,
                alternate_sign=False
            )
        else:  # Default to TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=2,
                max_df=0.95
            )
        
        self.vectorizer = vectorizer
        logger.info(f"Created {vectorizer_type} vectorizer with max_features={max_features}")
        return vectorizer
    
    def vectorize_texts(self, texts: Optional[List[str]] = None) -> np.ndarray:
        """
        Convert texts to feature vectors using scikit-learn.
        
        Args:
            texts: List of texts to vectorize, or None to use processed texts
            
        Returns:
            Feature matrix (sparse or dense)
        """
        if texts is None:
            texts = self.processed_texts if self.processed_texts else self.texts
        
        if self.vectorizer is None:
            self.create_vectorizer()
        
        feature_matrix = self.vectorizer.fit_transform(texts)
        logger.info(f"Vectorized {len(texts)} texts to shape {feature_matrix.shape}")
        return feature_matrix
    
    def apply_dimensionality_reduction(self, features: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to features.
        
        Args:
            features: Feature matrix to reduce
            
        Returns:
            Reduced feature matrix
        """
        if not self.data_config.get('apply_svd', True):
            return features
        
        n_components = self.data_config.get('svd_components', 100)
        
        # If n_components is too large relative to feature dimensions, adjust it
        if hasattr(features, 'shape'):
            if features.shape[1] <= n_components:
                n_components = max(2, features.shape[1] // 2)
                logger.warning(f"Adjusted SVD components to {n_components} " 
                               f"based on feature dimensions {features.shape}")
        
        # Create and fit SVD
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_features = self.svd.fit_transform(features)
        
        explained_variance = self.svd.explained_variance_ratio_.sum() * 100
        logger.info(f"Reduced features to {n_components} dimensions "
                   f"capturing {explained_variance:.2f}% of variance")
        
        return reduced_features
    
    def generate_encodings(self, texts: Optional[List[str]] = None) -> np.ndarray:
        """
        Generate numerical encodings for texts using scikit-learn pipeline.
        
        Args:
            texts: List of texts to encode, or None to use loaded texts
            
        Returns:
            Numerical encodings of texts
        """
        if texts is None:
            texts = self.texts
        
        # Preprocess texts
        processed_texts = self.preprocess_corpus(texts)
        
        # Create vectorizer if not already created
        if self.vectorizer is None:
            self.create_vectorizer()
        
        # Vectorize texts
        features = self.vectorize_texts(processed_texts)
        
        # Apply dimensionality reduction if configured
        if self.data_config.get('apply_svd', True):
            features = self.apply_dimensionality_reduction(features)
        
        # Normalize if needed
        if self.data_config.get('normalize', True):
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        
        self.encodings = features
        logger.info(f"Generated encodings with shape {features.shape}")
        return features
    
    def fill_missing_data(self, features: np.ndarray) -> np.ndarray:
        """
        Fill missing values in feature matrix using scikit-learn.
        
        Args:
            features: Feature matrix that may contain missing values
            
        Returns:
            Feature matrix with missing values filled
        """
        # Check if there are any missing values
        if isinstance(features, np.ndarray) and not np.isnan(features).any():
            return features
        
        strategy = self.data_config.get('imputation_strategy', 'mean')
        imputer = SimpleImputer(strategy=strategy)
        filled_features = imputer.fit_transform(features)
        
        logger.info(f"Filled missing values using {strategy} strategy")
        return filled_features
    
    def visualize_encodings(self, 
                           encodings: Optional[np.ndarray] = None,
                           labels: Optional[List[int]] = None,
                           save_path: str = 'data/visualization.png') -> None:
        """
        Create a visualization of encodings using t-SNE.
        
        Args:
            encodings: Encodings to visualize, or None to use stored encodings
            labels: Optional labels for coloring points
            save_path: Path to save the visualization
        """
        if encodings is None:
            encodings = self.encodings
        
        if encodings is None or len(encodings) == 0:
            logger.warning("No encodings available for visualization")
            return
        
        # Use t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(encodings)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            unique_labels = set(labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                idx = np.where(np.array(labels) == label)
                plt.scatter(reduced[idx, 0], reduced[idx, 1], c=[color], label=f'Cluster {label}')
            
            plt.legend()
        else:
            plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)
        
        plt.title('t-SNE Visualization of Text Encodings')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        
        # Save visualization
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Encoding visualization saved to {save_path}")
    
    def cluster_texts(self, n_clusters: int = 5) -> List[int]:
        """
        Cluster texts based on their encodings.
        
        Args:
            n_clusters: Number of clusters to create
            
        Returns:
            List of cluster labels for each text
        """
        if self.encodings is None:
            logger.warning("No encodings available for clustering")
            return []
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(self.encodings)
        
        # Count texts in each cluster
        cluster_counts = np.bincount(labels)
        for i, count in enumerate(cluster_counts):
            logger.info(f"Cluster {i}: {count} texts")
        
        # Visualize clusters
        self.visualize_encodings(self.encodings, labels, 'data/clusters.png')
        
        return labels.tolist()
    
    def save_vectorizer(self, path: str) -> None:
        """
        Save the vectorizer to disk.
        
        Args:
            path: Path to save the vectorizer
        """
        if self.vectorizer is None:
            logger.warning("No vectorizer to save")
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        logger.info(f"Vectorizer saved to {path}")
    
    def load_vectorizer(self, path: str) -> None:
        """
        Load the vectorizer from disk.
        
        Args:
            path: Path to the saved vectorizer
        """
        try:
            with open(path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            logger.info(f"Loaded vectorizer from {path}")
        except Exception as e:
            logger.error(f"Error loading vectorizer: {e}")
    
    def save_encodings(self, path: str) -> None:
        """
        Save encodings to disk.
        
        Args:
            path: Path to save the encodings
        """
        if self.encodings is None:
            logger.warning("No encodings to save")
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.encodings)
        
        logger.info(f"Encodings saved to {path}")
    
    def load_encodings(self, path: str) -> np.ndarray:
        """
        Load encodings from disk.
        
        Args:
            path: Path to the saved encodings
            
        Returns:
            Loaded encodings
        """
        try:
            self.encodings = np.load(path)
            logger.info(f"Loaded encodings from {path} with shape {self.encodings.shape}")
            return self.encodings
        except Exception as e:
            logger.error(f"Error loading encodings: {e}")
            return None
    
    def export_corpus_info(self, path: str) -> None:
        """
        Export corpus information to JSON.
        
        Args:
            path: Path to save the corpus info
        """
        if not self.texts:
            logger.warning("No corpus to export info for")
            return
        
        # Gather vocabulary info if available
        vocab_info = {}
        if hasattr(self.vectorizer, 'vocabulary_'):
            vocab = self.vectorizer.vocabulary_
            vocab_info = {
                'vocabulary_size': len(vocab),
                'top_words': sorted(vocab.items(), key=lambda x: x[1])[:20]
            }
        
        # Create corpus info dictionary
        corpus_info = {
            'num_texts': len(self.texts),
            'avg_text_length': sum(len(t.split()) for t in self.texts) / len(self.texts),
            'preprocessing': {
                'remove_stopwords': self.preprocessor.remove_stopwords,
                'lemmatize': self.preprocessor.lemmatize,
                'min_word_length': self.preprocessor.min_word_length
            },
            'vectorization': {
                'method': self.data_config.get('vectorizer', 'tfidf'),
                'max_features': self.data_config.get('max_features', 5000)
            },
            'dimensionality_reduction': {
                'applied': self.data_config.get('apply_svd', True),
                'n_components': self.data_config.get('svd_components', 100)
            },
            'vocabulary': vocab_info
        }
        
        # If SVD was applied, add explained variance
        if self.svd is not None:
            corpus_info['dimensionality_reduction']['explained_variance'] = float(
                self.svd.explained_variance_ratio_.sum()
            )
        
        # Save to file
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(corpus_info, f, indent=2)
        
        logger.info(f"Corpus info exported to {path}")
EOF

progress 45

# Create utils.py
echo -e "${BLUE}Creating utils.py...${NC}"
cat > utils.py << 'EOF'
"""
Utility functions for the GAN Tokenization System
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime
import datetime
import logging

logger = logging.getLogger('gan_tokenizer.utils')

def setup_logging(log_file='logs/gan_tokenizer.log', level=logging.INFO):
    """Set up logging configuration."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('gan_tokenizer')
    logger.info("Logging initialized")
    return logger

def setup_database(db_path='data/results.db'):
    """Create or connect to SQLite database for storing results."""
    engine = create_engine(f'sqlite:///{db_path}')
    metadata = MetaData()
    
    # Define tables if they don't exist
    results_table = Table('results', metadata,
        Column('id', Integer, primary_key=True),
        Column('timestamp', DateTime, default=datetime.datetime.utcnow),
        Column('encoding_id', String),
        Column('prompt', String),
        Column('response', String),
        Column('score', Float),
        Column('encoding_schema', String)
    )
    
    # Create tables
    metadata.create_all(engine)
    
    return engine, results_table

def visualize_encoding_schema(schema, save_path=None):
    """Visualize an encoding schema for better understanding."""
    if not schema:
        logger.warning("Empty schema provided")
        return
    
    # Count operation types
    op_types = {}
    for op in schema:
        op_type = op['type']
        op_types[op_type] = op_types.get(op_type, 0) + 1
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot operation type distribution
    plt.subplot(2, 2, 1)
    plt.bar(op_types.keys(), op_types.values())
    plt.title('Operation Types Distribution')
    plt.ylabel('Count')
    plt.grid(True, axis='y')
    
    # Plot operation sequence
    plt.subplot(2, 2, 2)
    op_colors = {'shift': 'blue', 'multiply': 'red', 'swap': 'green', 'transform': 'purple'}
    plt.title('Operation Sequence')
    for i, op in enumerate(schema):
        color = op_colors.get(op['type'], 'gray')
        plt.plot([i, i+1], [0, 0], color=color, linewidth=10, solid_capstyle='butt')
        
    plt.yticks([])
    plt.xlim(0, len(schema))
    plt.legend([plt.Line2D([0], [0], color=c, linewidth=10) for c in op_colors.values()], 
               op_colors.keys())
    
    # Display schema details
    plt.subplot(2, 1, 2)
    plt.axis('off')
    schema_text = "Schema Details:\n\n"
    for i, op in enumerate(schema):
        op_details = f"{i+1}. Type: {op['type']}"
        if op['type'] == 'shift':
            op_details += f", Value: {op['value']}"
        elif op['type'] == 'multiply':
            op_details += f", Value: {op['value']}"
        elif op['type'] == 'swap':
            op_details += f", Positions: {op['positions']}"
        elif op['type'] == 'transform':
            op_details += f", a: {op.get('a', 1)}, b: {op.get('b', 0)}"
        
        schema_text += op_details + "\n"
    
    plt.text(0, 0.9, schema_text, fontsize=10, va='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Schema visualization saved to {save_path}")
    else:
        plt.show()

def extract_top_schemas(db_path='data/results.db', top_n=5):
    """Extract the top performing encoding schemas."""
    engine = create_engine(f'sqlite:///{db_path}')
    
    # Query top results
    query = f"SELECT * FROM results ORDER BY score DESC LIMIT {top_n}"
    df = pd.read_sql(query, engine)
    
    if df.empty:
        logger.warning("No results found in database")
        return []
    
    top_schemas = []
    for _, row in df.iterrows():
        try:
            schema = json.loads(row['encoding_schema'])
            top_schemas.append({
                'id': row['encoding_id'],
                'score': float(row['score']),
                'schema': schema,
                'timestamp': row['timestamp']
            })
        except Exception as e:
            logger.error(f"Error parsing schema: {e}")
    
    # Save to file
    with open('data/top_schemas.json', 'w') as f:
        json.dump(top_schemas, f, indent=2)
    
    logger.info(f"Top {len(top_schemas)} schemas saved to data/top_schemas.json")
    return top_schemas

def generate_sentences(num=100, output_file='data/raw/generated_sentences.txt'):
    """Generate example sentences for testing."""
    topics = [
        "machine learning", "artificial intelligence", "neural networks", 
        "data science", "computer vision", "natural language processing",
        "deep learning", "reinforcement learning", "generative models",
        "transfer learning"
    ]
    
    structures = [
        "The {topic} approach enables researchers to {verb} {object}.",
        "Recent advances in {topic} have led to significant improvements in {object}.",
        "Using {topic} techniques, it is possible to {verb} {object} more efficiently.",
        "The field of {topic} focuses on developing methods to {verb} {object}.",
        "Researchers in {topic} aim to {verb} {object} through innovative algorithms.",
        "{topic} systems can effectively {verb} {object} without human intervention.",
        "The application of {topic} to {object} has shown promising results.",
        "State-of-the-art {topic} models can {verb} {object} with high accuracy.",
        "The integration of {topic} with {object} creates new opportunities for innovation.",
        "By leveraging {topic}, organizations can {verb} their {object} more effectively."
    ]
    
    verbs = [
        "analyze", "process", "optimize", "transform", "interpret",
        "classify", "predict", "generate", "enhance", "automate"
    ]
    
    objects = [
        "complex datasets", "unstructured data", "image recognition systems",
        "natural language understanding", "decision-making processes",
        "predictive models", "pattern recognition algorithms",
        "computational efficiency", "feature extraction methods",
        "high-dimensional data", "classification accuracy", "model performance"
    ]
    
    sentences = []
    for _ in range(num):
        topic = np.random.choice(topics)
        structure = np.random.choice(structures)
        verb = np.random.choice(verbs)
        obj = np.random.choice(objects)
        
        sentence = structure.format(topic=topic, verb=verb, object=obj)
        sentences.append(sentence)
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write('\n'.join(sentences))
    
    logger.info(f"Generated {num} sentences and saved to {output_file}")
    return sentences
EOF

progress 60

# Create main training script
echo -e "${BLUE}Creating gan_training.py...${NC}"
cat > gan_training.py << 'EOF'
"""
Enhanced GAN-based Text Encoding System
-------------------------------------
This script implements a GAN architecture to generate and refine text encoding schemas.
It uses scikit-learn for corpus management and a schema-based approach for model definition.
"""

import os
import sys
import time
import datetime
import json
import yaml
import random
import argparse
import logging
import psutil
import GPUtil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import TensorBoard
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import sqlalchemy as db
from tqdm import tqdm

# Import our custom modules
from schema_interpreter import GANSchemaInterpreter
from corpus_manager import CorpusManager, TextPreprocessor
from progress_visualization import ProgressVisualizer
from utils import setup_logging, extract_top_schemas, visualize_encoding_schema, setup_database

# Configure logging
logger = setup_logging()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Enhanced GAN-based Text Encoding System')
parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'generate'], 
                   help='Operation mode')
parser.add_argument('--data', type=str, default='data/raw/example_sentences.txt', 
                   help='Path to input text data')
parser.add_argument('--gpu', type=int, default=0, help='GPU device ID (-1 for CPU)')
parser.add_argument('--schema', type=str, default=None, 
                   help='Path to GAN schema file (overrides config)')
args = parser.parse_args()

# Load configuration
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Set TensorFlow GPU configuration
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0 and args.gpu >= 0:
    try:
        tf.config.set_visible_devices(physical_devices[args.gpu], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[args.gpu], True)
        logger.info(f"Using GPU {args.gpu}: {physical_devices[args.gpu]}")
    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")
else:
    logger.warning("No GPU available or CPU mode selected. This will be much slower.")

# Get schema file path
schema_file = args.schema if args.schema else config['model'].get('schema_file', 'schemas/default_gan_schema.yaml')

# Directories
LOG_DIR = os.path.join('logs', 'tensorboard')
MODEL_DIR = 'models'
DATA_DIR = 'data'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(MODEL_DIR, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)

# Create database engine for results
engine, results_table = setup_database('data/results.db')

class TextDataProcessor:
    """Handles text data loading, preprocessing, and encoding."""
    
    def __init__(self, config, corpus_manager=None):
        """Initialize with configuration."""
        self.config = config
        self.max_length = config['data']['max_sentence_length']
        self.embedding_dim = config['data']['embedding_dim']
        self.corpus_manager = corpus_manager
        self.sentences = []
        self.tokenizer = None
        self.vocab_size = 0
        
    def load_text_file(self, file_path):
        """Load text data from a file."""
        try:
            # If we have a corpus manager, use it
            if self.corpus_manager:
                self.sentences = self.corpus_manager.load_corpus(file_path)
                return self.sentences
            
            # Otherwise, use basic file loading
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Split text into sentences
            sentences = sent_tokenize(text)
            logger.info(f"Loaded {len(sentences)} sentences from {file_path}")
            
            # Filter out very short sentences
            sentences = [s for s in sentences if len(s.split()) >= 3]
            
            self.sentences = sentences
            return sentences
        except Exception as e:
            logger.error(f"Error loading text data: {e}")
            return []
    
    def create_vocabulary(self, min_count=2):
        """Create vocabulary from text data."""
        # If we have a corpus manager with scikit-learn vectorization
        if self.corpus_manager and self.config['data'].get('use_scikit', True):
            # Process and vectorize the text using scikit-learn
            self.corpus_manager.preprocess_corpus()
            self.corpus_manager.create_vectorizer()
            
            # Get vocabulary from vectorizer if available
            if hasattr(self.corpus_manager.vectorizer, 'vocabulary_'):
                vocab = self.corpus_manager.vectorizer.vocabulary_
                self.vocab_size = len(vocab) + 4  # Add special tokens
                logger.info(f"Using scikit-learn vocabulary with {self.vocab_size} tokens")
                
                # Create word-to-index mapping
                self.word2idx = {}
                self.idx2word = {}
                
                # Add special tokens
                special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
                for i, token in enumerate(special_tokens):
                    self.word2idx[token] = i
                    self.idx2word[i] = token
                
                # Add vocabulary words
                for word, idx in vocab.items():
                    self.word2idx[word] = idx + 4  # Offset for special tokens
                    self.idx2word[idx + 4] = word
                
                return self.word2idx
            
        # Fall back to traditional vocabulary creation
        # Tokenize sentences into words
        all_words = []
        for sentence in self.sentences:
            words = word_tokenize(sentence.lower())
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Create vocabulary (words that appear at least min_count times)
        vocabulary = {word for word, count in word_counts.items() if count >= min_count}
        
        # Add special tokens
        vocabulary = list(vocabulary)
        vocabulary = ['<PAD>', '<UNK>', '<START>', '<END>'] + vocabulary
        
        # Create word-to-index mapping
        self.word2idx = {word: idx for idx, word in enumerate(vocabulary)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(vocabulary)
        
        logger.info(f"Created traditional vocabulary with {self.vocab_size} tokens")
        return self.word2idx
    
    def encode_sentence(self, sentence, max_length=None):
        """Encode a sentence into numerical representation."""
        if max_length is None:
            max_length = self.max_length
            
        words = word_tokenize(sentence.lower())
        encoded = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        # Truncate or pad to max_length
        if len(encoded) > max_length:
            encoded = encoded[:max_length]
        else:
            encoded += [self.word2idx['<PAD>']] * (max_length - len(encoded))
            
        return np.array(encoded)
    
    def decode_sentence(self, encoded):
        """Decode numerical representation back to text."""
        words = [self.idx2word.get(idx, '<UNK>') for idx in encoded if idx != self.word2idx['<PAD>']]
        return ' '.join(words)
    
    def prepare_dataset(self, test_split=0.2):
        """Prepare dataset for training."""
        # If we have scikit-learn corpus management enabled
        if self.corpus_manager and self.config['data'].get('use_scikit', True):
            # Generate encodings with scikit-learn
            encodings = self.corpus_manager.generate_encodings()
            
            # We'll use these encodings as our "sentences"
            # But pad them to our required length
            padded_encodings = []
            for enc in encodings:
                # Normalize to 0-255 range for compatibility with our GAN
                normalized = (enc - enc.min()) / (enc.max() - enc.min() + 1e-8) * 255
                
                # Pad or truncate to max_length
                if len(normalized) > self.max_length:
                    padded = normalized[:self.max_length]
                else:
                    padded = np.pad(normalized, 
                                   (0, self.max_length - len(normalized)), 
                                   'constant', 
                                   constant_values=0)
                
                padded_encodings.append(padded)
            
            padded_encodings = np.array(padded_encodings)
            
            # Split into train and test sets
            indices = np.random.permutation(len(padded_encodings))
            test_size = int(len(padded_encodings) * test_split)
            test_indices = indices[:test_size]
            train_indices = indices[test_size:]
            
            train_data = padded_encodings[train_indices]
            test_data = padded_encodings[test_indices]
            
            logger.info(f"Prepared scikit-learn dataset with {len(train_data)} training and {len(test_data)} test samples")
            return train_data, test_data
        
        # Fall back to traditional encoding
        # Encode all sentences
        encoded_sentences = [self.encode_sentence(s) for s in self.sentences]
        encoded_sentences = np.array(encoded_sentences)
        
        # Split into train and test sets
        indices = np.random.permutation(len(encoded_sentences))
        test_size = int(len(encoded_sentences) * test_split)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        train_data = encoded_sentences[train_indices]
        test_data = encoded_sentences[test_indices]
        
        logger.info(f"Prepared traditional dataset with {len(train_data)} training and {len(test_data)} test samples")
        return train_data, test_data
    
    def apply_encoding_schema(self, sentence, encoding_schema):
        """Apply an encoding schema to transform a sentence."""
        # Handle both text sentences and numerical encodings
        if isinstance(sentence, str):
            encoded = self.encode_sentence(sentence)
        else:
            encoded = np.array(sentence)  # Already encoded
        
        # Apply each operation in the schema
        for operation in encoding_schema:
            op_type = operation['type']
            
            if op_type == 'shift':
                # Shift values by a constant
                encoded = (encoded + operation['value']) % self.vocab_size
            elif op_type == 'multiply':
                # Multiply by a constant (and mod by vocab size)
                encoded = (encoded * operation['value']) % self.vocab_size
            elif op_type == 'swap':
                # Swap positions
                pos1, pos2 = operation['positions']
                if pos1 < len(encoded) and pos2 < len(encoded):
                    encoded[pos1], encoded[pos2] = encoded[pos2], encoded[pos1]
            elif op_type == 'transform':
                # Apply a mathematical transformation
                for i in range(len(encoded)):
                    if encoded[i] != self.word2idx.get('<PAD>', 0):
                        x = encoded[i]
                        # Apply transformation function (e.g., x^2 + ax + b mod vocab_size)
                        degree = operation.get('degree', 2)
                        a = operation.get('a', 1)
                        b = operation.get('b', 0)
                        
                        if degree == 2:
                            encoded[i] = (x**2 + a*x + b) % self.vocab_size
                        elif degree == 3:
                            encoded[i] = (x**3 + a*x + b) % self.vocab_size
                        else:
                            encoded[i] = (x**degree) % self.vocab_size
        
        return encoded


class SchemaGAN:
    """GAN system for generating and refining text encoding schemas based on schema definition."""
    
    def __init__(self, data_processor, schema_interpreter, config):
        """Initialize with data processor, schema interpreter and configuration."""
        self.data_processor = data_processor
        self.schema_interpreter = schema_interpreter
        self.config = config
        self.progress_visualizer = ProgressVisualizer(config)
        
        # Define constants
        self.latent_dim = self.schema_interpreter.schema['gan_schema']['generator']['latent_dim']
        self.vocab_size = data_processor.vocab_size
        self.embedding_dim = config['data']['embedding_dim']
        self.max_length = config['data']['max_sentence_length']
        
        # Build generator and discriminator from schema
        self.build_models()
        
        # Set up TensorBoard
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(LOG_DIR, current_time)
        self.tensorboard = TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        
        # Set up LLM for evaluation
        self.setup_llm()
        
        # Training history
        self.history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'discriminator_accuracy': [],
            'llm_scores': []
        }
    
    def build_models(self):
        """Build generator and discriminator models from schema."""
        # Build generator
        self.generator = self.schema_interpreter.build_generator()
        
        # Build discriminator for encoded sentences
        self.discriminator = self.schema_interpreter.build_discriminator((self.max_length,))
        
        # Get optimizer from schema
        optimizer = self.schema_interpreter.get_optimizer()
        
        # Compile discriminator
        self.discriminator.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Build GAN
        self.discriminator.trainable = False
        self.gan_input = keras.Input(shape=(self.latent_dim,))
        
        # Generate schema and apply to sample sentence
        generated_encoding = self.generator(self.gan_input)
        
        # For the GAN model, we need a fixed sentence to apply the schema to
        # We'll represent this with a placeholder that will be replaced during training
        validity = self.discriminator(keras.Input(shape=(self.max_length,)))
        
        # Create the combined GAN model
        self.gan_model = keras.Model(self.gan_input, validity)
        self.gan_model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy'
        )
        
        logger.info("Models built successfully from schema")
        self.generator.summary()
        self.discriminator.summary()
        
        # Save model architectures
        self.schema_interpreter.save_model_architecture(
            self.generator, 
            os.path.join(MODEL_DIR, 'generator_architecture.json')
        )
        self.schema_interpreter.save_model_architecture(
            self.discriminator, 
            os.path.join(MODEL_DIR, 'discriminator_architecture.json')
        )
    
    def setup_llm(self):
        """Set up language model for encoding evaluation."""
        model_name = self.config['llm']['model_name']
        device = self.config['llm']['device']
        
        try:
            logger.info(f"Loading LLM: {model_name}")
            self.llm = pipeline(
                "text-generation",
                model=model_name,
                device=device
            )
            logger.info("LLM loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LLM: {e}")
            logger.warning("Will continue without LLM evaluation")
            self.llm = None
    
    def evaluate_encoding_with_llm(self, original_text, encoded_text, encoding_schema):
        """Evaluate the quality of an encoding using the LLM."""
        if self.llm is None:
            return {"score": 0.5, "feedback": "LLM not available"}
        
        prompt = f"""
        Analyze this text encoding schema:
        
        Original text: "{original_text}"
        Encoded representation: {encoded_text}
        Encoding schema: {json.dumps(encoding_schema, indent=2)}
        
        Evaluate the encoding on these criteria:
        1. Information preservation (how well can the original text be recovered)
        2. Compression efficiency
        3. Transformation complexity
        4. Potential for generalization to other texts
        
        On a scale of 0-100, rate this encoding schema.
        Return your analysis in JSON format with fields 'score' and 'feedback'.
        """
        
        try:
            response = self.llm(prompt, max_length=500, temperature=0.7)[0]['generated_text']
            
            # Extract JSON from response
            import re
            json_match = re.search(r'({.*?score.*?feedback.*?})', response, re.DOTALL)
            
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    # Normalize score to 0-1 range
                    result['score'] = float(result['score']) / 100.0
                    return result
                except json.JSONDecodeError:
                    pass
            
            # Fallback: extract score using regex
            score_match = re.search(r'score["\s:]+(\d+)', response, re.IGNORECASE)
            if score_match:
                score = float(score_match.group(1)) / 100.0
                return {"score": score, "feedback": response}
                
            return {"score": 0.5, "feedback": response}
        except Exception as e:
            logger.error(f"Error evaluating with LLM: {e}")
            return {"score": 0.5, "feedback": f"Error: {str(e)}"}
    
    def train(self, train_data, epochs=None, batch_size=None):
        """Train the GAN model."""
        if epochs is None:
            epochs = self.config['training']['epochs']
        if batch_size is None:
            batch_size = self.config['training']['batch_size']
            
        # Create directories for saving results
        os.makedirs(os.path.join(MODEL_DIR, 'generator'), exist_ok=True)
        os.makedirs(os.path.join(MODEL_DIR, 'discriminator'), exist_ok=True)
        
        save_interval = self.config['training']['save_interval']
        evaluation_interval = self.config['training']['evaluation_interval']
        
        # Start progress visualization
        self.progress_visualizer.start_training(epochs)
        
        logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Start epoch visualization
            steps_per_epoch = len(train_data) // batch_size
            self.progress_visualizer.start_epoch(epoch, steps_per_epoch)
            
            epoch_d_losses = []
            epoch_d_accs = []
            epoch_g_losses = []
            
            for batch_i in range(steps_per_epoch):
                # Train discriminator
                # Select random batch of sentences
                idx = np.random.randint(0, train_data.shape[0], batch_size)
                real_texts = train_data[idx]
                
                # Generate encoding schemas
                noise = tf.random.normal((batch_size, self.latent_dim))
                
                # Generate schemas and apply them to sentences
                generated_schemas_raw = self.generator(noise)
                fake_texts = np.zeros((batch_size, self.max_length))
                
                # Apply each schema to the corresponding real text
                for i in range(batch_size):
                    schema = self.schema_interpreter.interpret_encoding_operations(
                        generated_schemas_raw[i], 
                        self.vocab_size,
                        self.max_length
                    )
                    fake_texts[i] = self.data_processor.apply_encoding_schema(real_texts[i], schema)
                
                # Train discriminator
                d_loss_real = self.discriminator.train_on_batch(real_texts, np.ones((batch_size, 1)))
                d_loss_fake = self.discriminator.train_on_batch(fake_texts, np.zeros((batch_size, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # Train generator
                noise = tf.random.normal((batch_size, self.latent_dim))
                
                # When training the generator through the GAN model, we want the discriminator
                # to classify the generated texts as real (label=1)
                g_loss = self.gan_model.train_on_batch(noise, np.ones((batch_size, 1)))
                
                # Store batch losses
                epoch_d_losses.append(d_loss[0])
                epoch_d_accs.append(d_loss[1])
                epoch_g_losses.append(g_loss)
                
                # Update batch progress
                batch_losses = {'D_loss': d_loss[0], 'G_loss': g_loss}
                batch_metrics = {'D_acc': d_loss[1]}
                self.progress_visualizer.update_batch(batch_i, batch_losses, batch_metrics)
            
            # Calculate epoch losses (average of batch losses)
            epoch_d_loss = np.mean(epoch_d_losses)
            epoch_d_acc = np.mean(epoch_d_accs)
            epoch_g_loss = np.mean(epoch_g_losses)
            
            # Store losses in history
            self.history['generator_loss'].append(epoch_g_loss)
            self.history['discriminator_loss'].append(epoch_d_loss)
            self.history['discriminator_accuracy'].append(epoch_d_acc)
            
            # Evaluate with LLM periodically
            llm_feedback = None
            if epoch % evaluation_interval == 0:
                llm_feedback = self.evaluate_epoch(epoch, train_data)
            

            # End epoch visualization
            epoch_losses = {'D_loss': epoch_d_loss, 'G_loss': epoch_g_loss}
            epoch_metrics = {'D_acc': epoch_d_acc}
            self.progress_visualizer.end_epoch(epoch_losses, epoch_metrics, llm_feedback)
            
            # Save models periodically
            if epoch % save_interval == 0 or epoch == epochs - 1:
                self.save_models(epoch)
        
        # Plot training history
        self.plot_training_history()
        
        # Save final model
        self.save_models('final')
        
        # End training visualization
        self.progress_visualizer.end_training(
            {'D_loss': self.history['discriminator_loss'][-1], 'G_loss': self.history['generator_loss'][-1]},
            {'D_acc': self.history['discriminator_accuracy'][-1]}
        )
        
        logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
        
    def evaluate_epoch(self, epoch, train_data):
        """Evaluate the current model."""
        # Generate a sample schema
        noise = tf.random.normal((1, self.latent_dim))
        generated_schema_raw = self.generator.predict(noise)[0]
        schema = self.schema_interpreter.interpret_encoding_operations(
            generated_schema_raw, 
            self.vocab_size,
            self.max_length
        )
        
        # Choose a random sentence
        idx = np.random.randint(0, train_data.shape[0])
        original_enc = train_data[idx]
        
        # If using scikit-learn corpus, we need to handle differently
        if self.config['data'].get('use_scikit', True) and hasattr(self.data_processor, 'corpus_manager'):
            # For scikit-learn encodings, we don't have a direct text representation
            original_text = f"Encoding {idx}"
        else:
            # For traditional encoding, we can decode back to text
            original_text = self.data_processor.decode_sentence(original_enc)
        
        # Apply schema
        encoded_text = self.data_processor.apply_encoding_schema(original_enc, schema)
        
        # Evaluate with LLM
        result = self.evaluate_encoding_with_llm(original_text, encoded_text, schema)
        score = result['score']
        feedback = result['feedback']
        
        # Store the score
        self.history['llm_scores'].append(score)
        
        # Display LLM evaluation in progress visualization
        self.progress_visualizer.display_llm_evaluation(
            original_text, encoded_text, schema, result
        )
        
        # Log results
        logger.info(f"Epoch {epoch} evaluation:")
        logger.info(f"  Original text: {original_text}")
        logger.info(f"  Encoded text: {encoded_text}")
        logger.info(f"  Score: {score:.2f}")
        logger.info(f"  Feedback: {feedback[:100]}...")
        
        # Store in database
        with engine.connect() as conn:
            conn.execute(results_table.insert().values(
                timestamp=datetime.datetime.utcnow(),
                encoding_id=f"epoch_{epoch}",
                prompt=original_text,
                response=feedback,
                score=score,
                encoding_schema=json.dumps(schema)
            ))
            
        # Visualize schema
        visualize_encoding_schema(schema, os.path.join(LOG_DIR, f'schema_epoch_{epoch}.png'))
            
        return result
    
    def save_models(self, epoch):
        """Save model checkpoints."""
        generator_path = os.path.join(MODEL_DIR, 'generator', f'generator_epoch_{epoch}.h5')
        discriminator_path = os.path.join(MODEL_DIR, 'discriminator', f'discriminator_epoch_{epoch}.h5')
        
        self.generator.save(generator_path)
        self.discriminator.save(discriminator_path)
        
        # Save history
        history_path = os.path.join(MODEL_DIR, f'history_epoch_{epoch}.json')
        with open(history_path, 'w') as f:
            # Convert numpy types to Python native types for JSON serialization
            history_serializable = {}
            for key, values in self.history.items():
                history_serializable[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else v for v in values]
            
            json.dump(history_serializable, f, indent=2)
            
        logger.info(f"Models saved at epoch {epoch}")
    
    def plot_training_history(self):
        """Plot training losses and evaluation scores."""
        plt.figure(figsize=(12, 8))
        
        # Plot discriminator and generator losses
        plt.subplot(2, 1, 1)
        plt.plot(self.history['discriminator_loss'], label='Discriminator loss')
        plt.plot(self.history['generator_loss'], label='Generator loss')
        plt.title('GAN Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot discriminator accuracy
        plt.subplot(2, 2, 3)
        plt.plot(self.history['discriminator_accuracy'], label='Discriminator accuracy')
        plt.title('Discriminator Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot LLM evaluation scores
        if self.history['llm_scores']:
            plt.subplot(2, 2, 4)
            # Plot scores at their corresponding epochs
            eval_epochs = list(range(0, len(self.history['generator_loss']), 
                                     self.config['training']['evaluation_interval']))[:len(self.history['llm_scores'])]
            plt.plot(eval_epochs, self.history['llm_scores'], 'o-', label='LLM Evaluation Score')
            plt.title('LLM Evaluation Scores')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(LOG_DIR, 'training_history.png'))
        logger.info(f"Training history plot saved to {os.path.join(LOG_DIR, 'training_history.png')}")
    
    def generate_sample(self, num_samples=5):
        """Generate sample encoding schemas and apply them to texts."""
        # Load sample sentences if needed
        if not hasattr(self.data_processor, 'sentences') or not self.data_processor.sentences:
            self.data_processor.load_text_file(args.data)
        
        results = []
        for i in range(num_samples):
            # Generate a schema
            noise = tf.random.normal((1, self.latent_dim))
            generated_schema_raw = self.generator.predict(noise)[0]
            schema = self.schema_interpreter.interpret_encoding_operations(
                generated_schema_raw, 
                self.vocab_size,
                self.max_length
            )
            
            # Pick a random sentence
            # Handle both scikit-learn and traditional approaches
            if self.config['data'].get('use_scikit', True) and hasattr(self.data_processor, 'corpus_manager'):
                # For scikit-learn encodings
                if self.data_processor.corpus_manager.encodings is not None:
                    idx = np.random.randint(0, len(self.data_processor.corpus_manager.encodings))
                    encoding = self.data_processor.corpus_manager.encodings[idx]
                    # Normalize to 0-255 range for compatibility with our GAN
                    normalized = (encoding - encoding.min()) / (encoding.max() - encoding.min() + 1e-8) * 255
                    # Pad or truncate to max_length
                    if len(normalized) > self.max_length:
                        original_enc = normalized[:self.max_length]
                    else:
                        original_enc = np.pad(normalized, 
                                           (0, self.max_length - len(normalized)), 
                                           'constant', 
                                           constant_values=0)
                    sentence = f"Encoding {idx}"
                else:
                    # Fallback to a random sentence if no encodings
                    sentence = random.choice(self.data_processor.sentences)
                    original_enc = self.data_processor.encode_sentence(sentence)
            else:
                # Traditional approach
                sentence = random.choice(self.data_processor.sentences)
                original_enc = self.data_processor.encode_sentence(sentence)
            
            # Apply schema
            encoded = self.data_processor.apply_encoding_schema(original_enc, schema)
            
            # Try to decode back (only for traditional approach)
            if not self.config['data'].get('use_scikit', True) or not hasattr(self.data_processor, 'corpus_manager'):
                decoded = self.data_processor.decode_sentence(encoded)
            else:
                decoded = "Scikit-learn encoding (no direct text representation)"
            
            # Evaluate with LLM
            evaluation = self.evaluate_encoding_with_llm(sentence, encoded, schema)
            
            results.append({
                'sample_id': i,
                'original': sentence,
                'encoded': encoded.tolist(),
                'decoded': decoded,
                'schema': schema,
                'score': evaluation['score'],
                'feedback': evaluation['feedback']
            })
            
            # Display example
            self.progress_visualizer.display_example(
                original=sentence,
                encoded=encoded,
                decoded=decoded,
                score=evaluation['score']
            )
            
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Original: {sentence}")
            logger.info(f"  Encoded: {encoded.tolist()}")
            logger.info(f"  Decoded: {decoded}")
            logger.info(f"  Score: {evaluation['score']:.2f}")
            
            # Visualize schema
            visualize_encoding_schema(schema, os.path.join(DATA_DIR, f'schema_sample_{i}.png'))
            
        # Save results
        with open(os.path.join(DATA_DIR, 'samples.json'), 'w') as f:
            json.dump(results, f, indent=2)
            
        return results


def main():
    """Main function to run the enhanced GAN tokenizer system."""
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info(f"Starting Enhanced GAN Tokenizer run: {timestamp}")
    
    # Check GPU availability
    logger.info(f"TensorFlow version: {tf.__version__}")
    logger.info(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    
    # Load GAN schema
    schema_interpreter = GANSchemaInterpreter(schema_file)
    
    # Initialize corpus manager with scikit-learn capabilities
    corpus_manager = None
    if config['data'].get('use_scikit', True):
        corpus_manager = CorpusManager(config)
    
    # Initialize data processor
    data_processor = TextDataProcessor(config, corpus_manager)
    
    # Load text data
    data_processor.load_text_file(args.data)
    data_processor.create_vocabulary()
    
    # Prepare dataset
    train_data, test_data = data_processor.prepare_dataset(
        test_split=config['data']['test_split']
    )
    
    # Initialize SchemaGAN
    gan = SchemaGAN(data_processor, schema_interpreter, config)
    
    # Choose operation mode
    if args.mode == 'train':
        # Train the model
        gan.train(train_data)
    elif args.mode == 'evaluate':
        # Evaluate the model
        gan.evaluate_epoch('manual', train_data)
    elif args.mode == 'generate':
        # Generate sample encodings
        gan.generate_sample()
    
    logger.info(f"Enhanced GAN Tokenizer run completed: {timestamp}")


if __name__ == "__main__":
    main()
EOF

progress 75

echo -e "${GREEN}${BOLD}GAN network setup complete!${NC}"