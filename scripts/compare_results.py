import json
import sys

def compare(json1_path, json2_path):
    with open(json1_path, 'r', encoding='utf-8') as f1, \
         open(json2_path, 'r', encoding='utf-8') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    if len(data1) != len(data2):
        print(f"ВНИМАНИЕ: разное количество записей! {len(data1)} vs {len(data2)}")

    print(f"{'Файл':<35} | {'БЕЗ денойзинга':<40} | {'С денойзингом':<40} | {'Время (без/с)'}")
    print("-" * 130)
    
    same_count = 0
    diff_count = 0
    error_count = 0
    total_time_no = 0.0
    total_time_with = 0.0
    
    n = min(len(data1), len(data2))
    
    for i in range(n):
        file_name = data1[i]['file']
        text1 = data1[i].get('text', '')
        text2 = data2[i].get('text', '')
        time1 = data1[i].get('processing_sec', 0.0)
        time2 = data2[i].get('processing_sec', 0.0)
        err1 = data1[i].get('error', '')
        err2 = data2[i].get('error', '')
        
        total_time_no += time1
        total_time_with += time2
        
        if err1 or err2:
            error_count += 1
            status = "ОШИБКА"
        elif text1 == text2:
            same_count += 1
            status = "совпадает"
        else:
            diff_count += 1
            status = "различается"
        
        # Обрезаем текст для отображения
        text1_short = text1[:35] + "..." if len(text1) > 35 else text1
        text2_short = text2[:35] + "..." if len(text2) > 35 else text2
        
        print(f"{file_name:<35} | {text1_short:<40} | {text2_short:<40} | {time1:.2f}s / {time2:.2f}s | {status}")

    print("-" * 130)
    print(f"\nИТОГО: {n} файлов")
    print(f"  Совпадает текст: {same_count} ({same_count/n*100:.1f}%)")
    print(f"  Различается: {diff_count} ({diff_count/n*100:.1f}%)")
    print(f"  С ошибками: {error_count} ({error_count/n*100:.1f}%)")
    print(f"\nСреднее время обработки:")
    print(f"  Без денойзинга: {total_time_no/n:.2f} сек/файл")
    print(f"  С денойзингом:  {total_time_with/n:.2f} сек/файл")
    print(f"  Разница: {total_time_with/n - total_time_no/n:+.2f} сек/файл")

if __name__ == "__main__":
    json1 = sys.argv[1] if len(sys.argv) > 1 else "results/test_no_denoise.json"
    json2 = sys.argv[2] if len(sys.argv) > 2 else "results/test_with_denoise.json"
    compare(json1, json2)