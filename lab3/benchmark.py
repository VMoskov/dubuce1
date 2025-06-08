import subprocess
import time
import re
import sys
import pandas as pd
import csv


if __name__ == '__main__':
    start = time.time()

    --- BENCHMARKING BASELINE MODEL ---
    print('Benchmarking baseline model...')
    for i in range(5):
        model = 'baseline'
        output_file = f'results/{model}.log'
        command = [
            sys.executable, 
            'train.py',
            '--embedding_build_type', 'pretrained',
            '--model_type', model,
            '--embedding_dim', '300',
            '--hidden_dim', '100',
            '--num_layers', '1',
            '--train_batch_size', '10',
            '--test_batch_size', '32',
            '--num_epochs', '10',
            '--learning_rate', '1e-4',
            '--dropout', '0.0',
            '>>', output_file
        ]
        command_str = ' '.join(command)
        print(f'Running command: {command_str}')
        result = subprocess.run(command_str, shell=True, text=True)
        with open(output_file, 'a') as f:
            f.write('=' * 150)
            f.write('\n')

        time_elapsed = time.time() - start
        print(f'Time elapsed: {time_elapsed // 60} minutes {time_elapsed % 60} seconds')


    # --- BENCHMARKING DIFFERENT RNN MODELS CONFIGS ---
    print('Benchmarking different RNN models configurations...')
    all_results = []

    num_configs_per_model = 3*3*3*2  # hidden_size, num_layers, dropout, bidirectional
    total_configs = num_configs_per_model * 3  # 3 models: vanilla_rnn, gru, lstm

    tested_configs_per_model = 0
    tested_configs = 0

    for model in ['vanilla_rnn', 'gru', 'lstm']:
        tested_configs_per_model = 0
        for hidden_size in [100, 150, 200]:
            for num_layers in [1, 2, 3]:
                for dropout in [0.0, 0.25, 0.5]:
                    for bidirectional in [False, True]:
                        output_file = f'results/{model}.log'
                        test_config = f'{hidden_size=} {num_layers=} {dropout=} {bidirectional=}'

                        with open(output_file, 'a') as f:
                            f.write(f'Config: {test_config}\n')

                        command = [
                            sys.executable, 
                            'train.py',
                            '--embedding_build_type', 'pretrained',
                            '--model_type', model,
                            '--embedding_dim', '300',
                            '--hidden_dim', str(hidden_size),
                            '--num_layers', str(num_layers),
                            '--train_batch_size', '10',
                            '--test_batch_size', '32',
                            '--num_epochs', '10',
                            '--learning_rate', '1e-4',
                            '--dropout', str(dropout),
                            '--bidirectional' if bidirectional else '',
                            '>>', output_file
                        ]
                        command_str = ' '.join(command)
                        print(f'Running command: {command_str}')
                        result = subprocess.run(command_str, shell=True, text=True)

                        with open(output_file, 'r') as f:
                            lines = f.readlines()
                            last_line = lines[-1]

                        matches = re.findall(r'\d+\.\d+', last_line)
                        test_accuracy = float(matches[0])
                        test_f1 = float(matches[1])
                        print(f'Test accuracy: {test_accuracy}, Test F1: {test_f1}')
                        all_results.append({
                            'config': f'{model} {test_config}',
                            'test_accuracy': test_accuracy,
                            'test_f1': test_f1
                        })

                        with open(output_file, 'a') as f:
                            f.write('=' * 150)
                            f.write('\n')

                        tested_configs_per_model += 1
                        tested_configs += 1
                        time_elapsed = time.time() - start
                        print(f'Tested {tested_configs_per_model}/{num_configs_per_model} configs for {model}')
                        print(f'Total tested configs: {tested_configs}/{total_configs}')
                        print(f'Time elapsed: {time_elapsed // 60} minutes {time_elapsed % 60} seconds')

    print('All configurations tested.')
    print()
    print()

    all_results_df = pd.DataFrame(all_results) 
    all_results_df = all_results_df.sort_values(by='test_f1', ascending=False)
    all_results_df.to_csv('results/benchmark_results.csv', index=False)

    # --- BENCHMARKING BEST RNN MODEL ---
    print('Benchmarking best RNN model and baseline model without pretrained embeddings...')
    with open('results/benchmark_results.csv', 'r') as f:
        first_line = f.readline().strip()  # contains header
        first_row = f.readline().strip()  # contains first row of data

    best_config = first_row.split(',')[0]  # get the first column which is the config

    parts = best_config.split(' ')
    best_model = parts[0]
    hidden_size = int(parts[1].split('=')[1])
    num_layers = int(parts[2].split('=')[1])
    dropout = float(parts[3].split('=')[1])
    bidirectional = parts[4].split('=')[1] == 'True'

    # run the best model without pretrained embeddings
    random_init_results = []
    output_file = f'results/random_init.log'
    for model in [best_model, 'baseline']:
        command = [
            sys.executable, 
            'train.py',
            '--embedding_build_type', 'random',
            '--model_type', model,
            '--embedding_dim', '300',
            '--hidden_dim', str(hidden_size),
            '--num_layers', str(num_layers),
            '--train_batch_size', '10',
            '--test_batch_size', '32',
            '--num_epochs', '10',
            '--learning_rate', '1e-4',
            '--dropout', str(dropout),
            '--bidirectional' if bidirectional else '',
            '>>', output_file
        ]
        command_str = ' '.join(command)
        print(f'Running command: {command_str}')
        result = subprocess.run(command_str, shell=True, text=True)

        with open(output_file, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]

        matches = re.findall(r'\d+\.\d+', last_line)
        test_accuracy = float(matches[0])
        test_f1 = float(matches[1])
        random_init_results.append({
            'model': model,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1
        })

        time_elapsed = time.time() - start
        print(f'Time elapsed: {time_elapsed // 60} minutes {time_elapsed % 60} seconds')

    random_init_df = pd.DataFrame(random_init_results)
    random_init_df = random_init_df.sort_values(by='test_f1', ascending=False)
    random_init_df.to_csv('results/random_init_results.csv', index=False)
    print('Random init embeddings benchmark completed.')
    print()
    print()

    # --- BENCHMARKING BEST RNN MODEL ON OTHER HYPERPARAMETERS ---
    print('Benchmarking best RNN model on other hyperparameters...')
    other_hyperparams_results = []
    output_file = f'results/{best_model}_hparams_optimization.log'

    total_configs = 3*3*3*3*3
    tested_configs = 0
    for vocab_size in [1000, 2000, 3000]:
        for batch_size in [10, 20, 32]:
            for dropout in [0.0, 0.25, 0.5]:
                for num_layers in [1, 2, 3]:
                    for hidden_size in [100, 150, 200]:
                        train_batch_size = batch_size
                        test_batch_size = int(batch_size * 1.5)
                        test_config = f'{vocab_size=} {train_batch_size=} {test_batch_size=} {dropout=} {num_layers=} {hidden_size=}'

                        with open(output_file, 'a') as f:
                            f.write(f'Config: {test_config}\n')

                        command = [
                            sys.executable, 
                            'train.py',
                            '--embedding_build_type', 'pretrained',
                            '--model_type', best_model,
                            '--embedding_dim', '300',
                            '--hidden_dim', str(hidden_size),
                            '--num_layers', str(num_layers),
                            '--vocab_size', str(vocab_size),
                            '--train_batch_size', str(train_batch_size),
                            '--test_batch_size', str(test_batch_size),
                            '--num_epochs', '10',
                            '--learning_rate', '1e-4',
                            '--dropout', str(dropout),
                            '--bidirectional' if bidirectional else '',
                            '>>', output_file
                        ]

                        command_str = ' '.join(command)
                        print(f'Running command: {command_str}')
                        result = subprocess.run(command_str, shell=True, text=True)

                        with open(output_file, 'r') as f:
                            lines = f.readlines()
                            last_line = lines[-1]

                        matches = re.findall(r'\d+\.\d+', last_line)
                        test_accuracy = float(matches[0])
                        test_f1 = float(matches[1])
                        print(f'Test accuracy: {test_accuracy}, Test F1: {test_f1}')
                        other_hyperparams_results.append({
                            'config': f'{best_model} {test_config}',
                            'test_accuracy': test_accuracy,
                            'test_f1': test_f1
                        })

                        with open(output_file, 'a') as f:
                            f.write('=' * 150)
                            f.write('\n')

                        tested_configs += 1
                        time_elapsed = time.time() - start
                        print(f'Tested {tested_configs}/{total_configs} configs')
                        print(f'Time elapsed: {time_elapsed // 60} minutes {time_elapsed % 60} seconds')

    print('All hyperparameters configurations tested.')
    other_hyperparams_df = pd.DataFrame(other_hyperparams_results)
    other_hyperparams_df = other_hyperparams_df.sort_values(by='test_f1', ascending=False)
    other_hyperparams_df.to_csv(f'results/{best_model}_hparams_optimization.csv', index=False)
    print()
    print()

    # --- BENCHMARKING BEST RNN CONFIGS WITH ATTENTION ---
    print('Benchmarking best RNN configs with attention...')
    attention_results = []
    commands = []

    output_file = f'results/attention.log'
    with open('results/benchmark_results.csv', 'r') as f:
        first_line = f.readline().strip()  # contains header
        # take first 10 rows
        for i in range(10):
            row = f.readline().strip()
            config = row.split(',')[0]  # get the first column which is the config
            parts = config.split(' ')
            model = parts[0]
            hidden_size = int(parts[1].split('=')[1])
            num_layers = int(parts[2].split('=')[1])
            dropout = float(parts[3].split('=')[1])
            bidirectional = parts[4].split('=')[1] == 'True'
            for attention in [True, False]:
                command = [
                    sys.executable, 
                    'train.py',
                    '--embedding_build_type', 'pretrained',
                    '--model_type', model,
                    '--embedding_dim', '300',
                    '--hidden_dim', str(hidden_size),
                    '--num_layers', str(num_layers),
                    '--train_batch_size', '10',
                    '--test_batch_size', '32',
                    '--num_epochs', '10',
                    '--learning_rate', '5e-5',
                    '--dropout', str(dropout),
                    '--bidirectional' if bidirectional else '',
                    '--attention' if attention else '',
                    '>>', output_file
                ]
                command_str = ' '.join(command)
                commands.append({
                    'command': command_str,
                    'config': f'{model} {hidden_size=} {num_layers=} {dropout=} {bidirectional=} {attention=}'
                })
    
    with open('results/gru_hparams_optimization.csv', 'r') as f:
        first_line = f.readline().strip()
        # take first 10 rows
        for i in range(10):
            row = f.readline().strip()
            config = row.split(',')[0]
            parts = config.split(' ')
            model = parts[0]
            vocab_size = int(parts[1].split('=')[1])
            train_batch_size = int(parts[2].split('=')[1])
            test_batch_size = int(parts[3].split('=')[1])
            dropout = float(parts[4].split('=')[1])
            num_layers = int(parts[5].split('=')[1])
            hidden_size = int(parts[6].split('=')[1])
            for attention in [True, False]:
                command = [
                    sys.executable, 
                    'train.py',
                    '--embedding_build_type', 'pretrained',
                    '--model_type', model,
                    '--embedding_dim', '300',
                    '--hidden_dim', str(hidden_size),
                    '--num_layers', str(num_layers),
                    '--vocab_size', str(vocab_size),
                    '--train_batch_size', str(train_batch_size),
                    '--test_batch_size', str(test_batch_size),
                    '--num_epochs', '10',
                    '--learning_rate', '5e-5',
                    '--dropout', str(dropout),
                    '--bidirectional' if bidirectional else '',
                    '--attention' if attention else '',
                    '>>', output_file
                ]
                command_str = ' '.join(command)
                commands.append({
                    'command': command_str,
                    'config': f'{model} {vocab_size=} {train_batch_size=} {test_batch_size=} {dropout=} {num_layers=} {hidden_size=} {attention=}'
                })

    tested_configs = 0
    total_configs = len(commands)
    for idx, cmd in enumerate(commands):
        command = cmd['command']
        print(f'Running command: {command}')
        result = subprocess.run(command, shell=True, text=True)

        with open(output_file, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]

        matches = re.findall(r'\d+\.\d+', last_line)
        test_accuracy = float(matches[0])
        test_f1 = float(matches[1])
        print(f'Test accuracy: {test_accuracy}, Test F1: {test_f1}')
        attention_results.append({
            'config': cmd['config'],
            'test_accuracy': test_accuracy,
            'test_f1': test_f1
        })

        with open(output_file, 'a') as f:
            f.write('=' * 150)
            f.write('\n')

        tested_configs += 1
        time_elapsed = time.time() - start
        print(f'Tested {idx + 1}/{len(commands)} configs')
        print(f'Time elapsed: {time_elapsed // 60} minutes {time_elapsed % 60} seconds')

    print('All attention configurations tested.')
    attention_df = pd.DataFrame(attention_results)
    attention_df = attention_df.sort_values(by='test_f1', ascending=False)
    attention_df.to_csv(f'results/attention_results.csv', index=False)
    print('Attention benchmark completed.')
    print()