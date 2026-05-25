def read_first_n_lines(input_file, output_file, n=10):
    """
    读取文件前n行，存入到新文件中
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        n: 读取的行数，默认10行
    """
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for i, line in enumerate(f_in):
                if i >= n:
                    break
                f_out.write(line)

# 使用示例
read_first_n_lines('data/test.json', 'data/test1.json', n=1487)