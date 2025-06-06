import subprocess
import sys


if __name__ == '__main__':
    for i in range(5):
        output_file = f'baseline.log'
        command = [
            sys.executable, 
            'train.py',
            '--embedding_build_type', 'random',
            '--model_type', 'baseline',
            '--embedding_dim', '100',
            '--hidden_dim', '100',
            '--num_layers', '1',
            '--train_batch_size', '32',
            '--test_batch_size', '32',
            '--num_epochs', '10',
            '--learning_rate', '0.001',
            '--dropout', '0.0',
            '--bidirectional', 'False',
            '>>', output_file
        ]
        command_str = ' '.join(command)
        print(f'Running command: {command_str}')
        result = subprocess.run(command_str, shell=True, text=True)
        with open(output_file, 'a') as f:
            f.write('=' * 150)
            f.write('\n')


    for model in ['vanilla_rnn', 'gru', 'lstm']:
        for hidden_size in [100, 150, 200]:
            for num_layers in [1, 2, 3]:
                for dropout in [0.0, 0.25, 0.5]:
                    for bidirectional in [False, True]:
                        output_file = f'results/{model}_hidden={hidden_size}_layers={num_layers}_dropout={dropout}_bidirectional={bidirectional}.log'
                        command = [
                            sys.executable, 
                            'train.py',
                            '--embedding_build_type', 'random',
                            '--model_type', model,
                            '--embedding_dim', '100',
                            '--hidden_dim', str(hidden_size),
                            '--num_layers', str(num_layers),
                            '--train_batch_size', '32',
                            '--test_batch_size', '32',
                            '--num_epochs', '10',
                            '--learning_rate', '0.001',
                            '--dropout', str(dropout),
                            '--bidirectional' if bidirectional else '',
                            '>>', output_file
                        ]
                        command_str = ' '.join(command)
                        print(f'Running command: {command_str}')
                        result = subprocess.run(command_str, shell=True, text=True)
                        with open(output_file, 'a') as f:
                            f.write('=' * 150)
                            f.write('\n')