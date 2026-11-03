import csv
import sys
import re

def transform_csv(input_file, output_file):
    try:
        with open(input_file, 'r', newline='') as infile:
            lines = infile.readlines()
        
        current_prefix = None  # Текущий активный префикс
        current_index = 1      # Счетчик строк, сбрасывается после каждого 'day'
        output_lines = []      # Результирующие строки для записи
        processed_any = False  # Флаг, что были обработаны новые данные
        
        # Обработка строк по порядку
        for line in lines:
            original_line = line.strip()
            if not original_line:
                output_lines.append("")  # Сохраняем пустые строки
                continue
            
            # Проверка, является ли строка уже отформатированной (x,x,x)
            if re.match(r'^[^,]+,[^,]+,[^,]+$', original_line):
                # Строка уже в нужном формате - пропускаем обработку
                output_lines.append(original_line)
                continue
            
            # Проверка на строку с "day" для обновления префикса и сброса индекса
            if "day" in original_line.lower():
                # Извлекаем последнее число из строки как префикс
                numbers = re.findall(r'[-+]?\d*\.\d+|\d+', original_line)
                if numbers:
                    try:
                        current_prefix = float(numbers[-1])
                        current_index = 1  # Сбрасываем нумерацию
                        # Не добавляем саму строку 'day' в результат
                        continue
                    except ValueError:
                        pass
            
            # Попытка преобразовать строку в число
            try:
                value = float(original_line)
                # Обрабатываем только если есть текущий префикс
                if current_prefix is not None:
                    offset = -0.0207  # Смещение для получения целевых значений
                    adjusted_value = value + offset
                    adjusted_value = round(adjusted_value, 4)
                    
                    # Форматируем строку в нужном формате с префиксом в 1 знак после точки
                    formatted_line = f"{current_prefix:.1f},{current_index},{adjusted_value:.4f}"
                    output_lines.append(formatted_line)
                    current_index += 1
                    processed_any = True
                    continue
                # Если префикс не установлен, сохраняем исходную строку
                output_lines.append(original_line)
            except ValueError:
                # Не числовая строка и не 'day' - сохраняем как есть
                output_lines.append(original_line)
        
        # Запись результата в выходной файл
        with open(output_file, 'w', newline='') as outfile:
            for line in output_lines:
                outfile.write(line + '\n')
        
        # Статистика обработки
        print(f"Файл успешно обработан: {input_file} -> {output_file}")
        if processed_any:
            print("Новые данные обработаны и добавлены в файл")
        else:
            print("Новых данных для обработки не найдено")
        
        # Подсчет статистики по секциям
        section_stats = {}
        current_section = None
        line_count = 0
        
        for line in output_lines:
            if not line:
                continue
            
            # Определяем начало новой секции по префиксу в отформатированных строках
            if re.match(r'^\d+\.\d,\d+,\d+\.\d+$', line):  # Формат x.x,x,xxxx.xxxx
                parts = line.split(',')
                try:
                    prefix = float(parts[0])
                    if prefix != current_section:
                        if current_section is not None:
                            section_stats[current_section] = section_stats.get(current_section, 0) + line_count
                        current_section = prefix
                        line_count = 0
                    line_count += 1
                except (ValueError, IndexError):
                    continue
        
        # Добавляем последнюю секцию
        if current_section is not None:
            section_stats[current_section] = section_stats.get(current_section, 0) + line_count
        
        # Вывод статистики с форматированием префикса в 1 знак
        if section_stats:
            print("\nСтатистика по секциям:")
            for prefix, count in sorted(section_stats.items()):
                print(f"  Секция с префиксом {prefix:.1f}: {count} строк")
        
        return True
    
    except FileNotFoundError:
        print(f"Ошибка: файл '{input_file}' не найден")
        return False
    except PermissionError:
        print(f"Ошибка: нет прав на запись в файл '{output_file}'")
        return False
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Использование: python script.py <входной_файл.csv> <выходной_файл.csv>")
        print("Пример: python script.py input.csv output.csv")
        sys.exit(1)
    
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    
    success = transform_csv(input_filename, output_filename)
    sys.exit(0 if success else 1)