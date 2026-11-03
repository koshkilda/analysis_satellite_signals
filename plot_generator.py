#!/usr/bin/env python3
import matplotlib

matplotlib.use('Agg')
import sys
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import warnings
import argparse
import pathlib
from typing import Optional, Dict
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================
# Цвета для разных кластеров (по индексу моды)
MODE_COLORS: Dict[int, str] = {
    0: '#1f77b4', 1: '#22d3ee', 2: '#22c55e', 3: '#eab308',
    4: '#f97316', 5: '#ef4444', 6: '#a855f7', 7: '#ec4899',
}

# Цвета в зависимости от количества мод (для 2D и 3D)
MODE_COUNT_COLORS: Dict[int, str] = {
    1: '#1f77b4',   # унимодальный -> синий
    2: '#22d3ee',   # бимодальный -> голубой
}

# Параметры оформления
PLOT_DPI = 200
LINE_WIDTH = 2.5
FILL_ALPHA = 0.15
GRID_ALPHA = 0.6
LEGEND_FS = 11
LABEL_FS = 14
TITLE_FS = 16
FIG_2D = (14, 7)
FIG_3D = (20, 15)
MIN_POINTS_FOR_LINE = 3
BIN_WIDTH = 120.0          # ширина бина для агрегации (сек)
CLUSTER_PADDING = 0.5      # минимальный паддинг для 3D графика

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================
def get_mode_color(mode_idx: int) -> str:
    """Возвращает цвет для индекса моды/кластера."""
    return MODE_COLORS.get(mode_idx % len(MODE_COLORS), MODE_COLORS[0])

def load_csv(path: pathlib.Path) -> Optional[pd.DataFrame]:
    """Загружает CSV-файл, возвращает None при ошибке."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, na_values=['NaN', 'nan', ''], comment='#')
        return df if not df.empty else None
    except Exception:
        return None

def read_meta_from_header(path: pathlib.Path) -> Dict[str, float]:
    """Извлекает метаданные из первой строки-комментария CSV."""
    meta = {}
    with open(path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if first_line.startswith('#'):
            content = first_line[1:].strip()
            for item in content.split(','):
                try:
                    key, value = item.strip().split('=')
                    meta[key.strip()] = float(value.strip())
                except (ValueError, IndexError):
                    continue
    return meta

def valid_error_mask(series: pd.Series) -> pd.Series:
    """Создаёт маску для валидных значений ошибок (не пустые и положительные)."""
    return series.notna() & (series > 0)

def add_confidence_band(ax, x_data: pd.Series, mean_data: pd.Series,
                        std_data: pd.Series, color: str) -> Optional[float]:
    valid_std = std_data.notna() & (std_data > 0)
    if not valid_std.any():
        return None
    
    x_valid = x_data[valid_std]
    mean_valid = mean_data[valid_std]
    std_valid = std_data[valid_std]
    
    ax.fill_between(
        x_valid,
        mean_valid - std_valid,
        mean_valid + std_valid,
        color=color,
        alpha=FILL_ALPHA,
        zorder=4
    )
    
    return float(std_valid.mean())

def compute_cluster_counts_per_bin(df: pd.DataFrame, bin_width: float = BIN_WIDTH) -> Dict[float, int]:
    """
    Возвращает словарь {время_бина: количество_уникальных_кластеров}.
    Время бина соответствует левой границе интервала.
    """
    df_temp = df.copy()
    t_min, t_max = df_temp['time'].min(), df_temp['time'].max()
    bins = np.arange(t_min, t_max + bin_width, bin_width)
    df_temp['time_bin'] = pd.cut(df_temp['time'], bins=bins, labels=bins[:-1]).astype(float)
    return df_temp.groupby('time_bin')['window_cluster'].nunique().to_dict()


# ============================================================================
# 2D ГРАФИК
# ============================================================================
def detect_mode_transitions(df: pd.DataFrame, time_col: str = 'time',
                            cluster_col: str = 'window_cluster') -> Optional[float]:
    """Определяет приблизительное время, когда появляется новая мода."""
    if cluster_col not in df.columns or time_col not in df.columns:
        return None

    df_sorted = df.sort_values(time_col).copy()
    unique_times = df_sorted[time_col].unique()
    if len(unique_times) < 10:
        return None

    initial_clusters = df_sorted[df_sorted[time_col] == unique_times[0]][cluster_col].unique()
    for t_val in unique_times[1:]:
        current_clusters = df_sorted[df_sorted[time_col] == t_val][cluster_col].unique()
        new_clusters = set(current_clusters) - set(initial_clusters)
        if new_clusters:
            return t_val
    return None

def aggregate_by_mode(df: pd.DataFrame, time_col: str = 'time',
                      error_col: str = 'error', cluster_col: str = 'window_cluster',
                      bin_width: float = BIN_WIDTH) -> Dict[int, pd.DataFrame]:
    """Агрегирует статистику ошибок по модам и временным бинам."""
    if cluster_col not in df.columns:
        return {}

    result = {}
    df_valid = df[valid_error_mask(df[error_col])].copy()
    if df_valid.empty:
        return result

    t_min, t_max = df_valid[time_col].min(), df_valid[time_col].max()
    bins = np.arange(t_min, t_max + bin_width, bin_width)
    df_valid['time_bin'] = pd.cut(df_valid[time_col], bins=bins, labels=bins[:-1])

    for cluster_id in df_valid[cluster_col].unique():
        cluster_data = df_valid[df_valid[cluster_col] == cluster_id]
        if len(cluster_data) < MIN_POINTS_FOR_LINE:
            continue

        agg = cluster_data.groupby('time_bin')[error_col].agg(['mean', 'std', 'count']).reset_index()
        agg = agg[agg['count'] >= 2]
        agg['time_bin'] = agg['time_bin'].astype(float)
        agg = agg.rename(columns={'time_bin': time_col,
                                  'mean': f'{error_col}_mean',
                                  'std': f'{error_col}_std'})
        if not agg.empty:
            result[int(cluster_id)] = agg

    return result

def plot_2d_unified(csv_path: str, out_dir: str, split_override: Optional[float] = None) -> bool:
    """2D график: непрерывные линии кластеров, цвет = количество мод."""
    df = load_csv(pathlib.Path(csv_path))
    if df is None:
        print(f"Не удалось загрузить CSV: {csv_path}")
        return False

    required_cols = ['time', 'error', 'window_cluster']
    if not all(col in df.columns for col in required_cols):
        print(f"Отсутствуют требуемые колонки. Найдено: {list(df.columns)}")
        return False

    split_time = split_override or detect_mode_transitions(df)
    cluster_counts = compute_cluster_counts_per_bin(df, BIN_WIDTH)
    mode_stats = aggregate_by_mode(df, bin_width=BIN_WIDTH)

    if not mode_stats:
        print("Нет валидных данных для построения графика после агрегации")
        return False

    fig, ax = plt.subplots(figsize=FIG_2D, dpi=PLOT_DPI)

    ci_half_widths = []

    for mode_idx, stats_df in sorted(mode_stats.items()):
        if len(stats_df) < 2:
            continue
        stats_df = stats_df.sort_values('time')
        times = stats_df['time'].values
        means = stats_df['error_mean'].values

        segments = []
        colors = []
        for i in range(len(times) - 1):
            n_clust = cluster_counts.get(times[i], 1)
            color = MODE_COUNT_COLORS.get(n_clust, MODE_COUNT_COLORS[1])
            segments.append([(times[i], means[i]), (times[i + 1], means[i + 1])])
            colors.append(color)

        lc = LineCollection(segments, colors=colors, linewidth=LINE_WIDTH, alpha=0.9, zorder=5)
        ax.add_collection(lc)

        if 'error_std' in stats_df.columns:
            first_n_clust = cluster_counts.get(times[0], 1)
            main_color = MODE_COUNT_COLORS.get(first_n_clust, MODE_COUNT_COLORS[1])
            half_width = add_confidence_band(
                ax, stats_df['time'], stats_df['error_mean'],
                stats_df['error_std'], main_color
            )
            if half_width is not None:
                ci_half_widths.append(half_width)

    # Подпись о ширине доверительной полосы на самом графике (левый нижний угол)
    if ci_half_widths:
        avg_hw = sum(ci_half_widths) / len(ci_half_widths)
        ax.text(
            0.02, 0.02,
            f'Доверительная полоса: ±{avg_hw:.2f} м (среднее σ)',
            transform=ax.transAxes, fontsize=9, color='#334155',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )

    if split_time is not None:
        ax.annotate(f'split time = {split_time:.0f} с',
                    xy=(0.98, 0.98), xycoords='axes fraction',
                    ha='right', va='top', fontsize=10, color='#ef4444', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.grid(True, linestyle='--', alpha=GRID_ALPHA, color='#94a3b8')
    ax.set_xlabel('Время (с)', fontsize=LABEL_FS, fontweight='bold')
    ax.set_ylabel('Ошибка псевдодальности (м)', fontsize=LABEL_FS, fontweight='bold')
    ax.set_title('Эволюция ошибок по кластерам (цвет = число мод)', fontsize=TITLE_FS, fontweight='bold', pad=15)

    legend_elements = [
        Patch(facecolor=MODE_COUNT_COLORS[1], label='Унимодальный режим (1 мода)'),
        Patch(facecolor=MODE_COUNT_COLORS[2], label='Бимодальный режим (2 моды)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=LEGEND_FS, frameon=True, fancybox=True)

    all_errors = pd.concat([s['error_mean'] for s in mode_stats.values()], ignore_index=True)
    if not all_errors.empty:
        err_min, err_max = all_errors.min(), all_errors.max()
        padding = max((err_max - err_min) * 0.15, 2.0)
        ax.set_ylim(max(0, err_min - padding), err_max + padding)

    time_range = df['time'].max() - df['time'].min()
    ax.set_xlim(df['time'].min() - time_range * 0.02, df['time'].max() + time_range * 0.02)

    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path / '2d_plot.png', dpi=PLOT_DPI, bbox_inches='tight')
    plt.savefig(out_path / '2d_plot.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"2D график сохранён в {out_path}")
    return True

# ============================================================================
# 3D ГРАФИК
# ============================================================================
def plot_3d_unified(csv_path: str, out_dir: str, max_time_slices: int = 50) -> bool:
    """3D график эволюции плотности ошибок. Цвет = количество мод."""
    fp = pathlib.Path(csv_path)
    df = load_csv(fp)
    if df is None:
        print(f"Не удалось загрузить CSV: {csv_path}")
        return False

    required_cols = ['time', 'error', 'density', 'window_cluster']
    if not all(col in df.columns for col in required_cols):
        print(f"Отсутствуют колонки для 3D. Найдено: {list(df.columns)}")
        return False

    # Отбрасываем нулевые плотности
    df = df[df['density'] > 1e-8].copy()
    if df.empty:
        print("Нет данных с ненулевой плотностью")
        return False

    meta = read_meta_from_header(fp)

    # Выбираем временные срезы
    unique_times = np.sort(df['time'].unique())
    if len(unique_times) > max_time_slices:
        step = len(unique_times) // max_time_slices
        selected_times = unique_times[::step]
    else:
        selected_times = unique_times

    if len(selected_times) < 2:
        print("Недостаточно временных точек")
        return False

    # Общая сетка ошибок
    error_min = meta.get('minError', df['error'].min())
    error_max = meta.get('maxError', df['error'].max())
    error_grid = np.linspace(error_min, error_max, 200)

    fig = plt.figure(figsize=FIG_3D)
    ax = fig.add_subplot(111, projection='3d')

    for t_val in selected_times:
        slice_data = df[df['time'] == t_val].sort_values('error')
        if len(slice_data) < 3:
            continue

        n_modes = slice_data['window_cluster'].nunique()
        color = MODE_COUNT_COLORS.get(n_modes, MODE_COUNT_COLORS[1])

        density_total = np.zeros_like(error_grid)

        for _, group in slice_data.groupby('window_cluster'):
            if len(group) < 2:
                continue

            err_min = group['error'].min()
            err_max = group['error'].max()
            pad = max((err_max - err_min) * 0.05, CLUSTER_PADDING)
            local_min = err_min - pad
            local_max = err_max + pad

            local_grid = np.linspace(local_min, local_max, 150)
            local_density = np.interp(
                local_grid,
                group['error'].values,
                group['density'].values,
                left=0.0,
                right=0.0
            )

            mask = (error_grid >= local_min) & (error_grid <= local_max)
            if np.any(mask):
                density_total[mask] = np.interp(
                    error_grid[mask],
                    local_grid,
                    local_density,
                    left=0.0,
                    right=0.0
                )

        ax.plot(
            [t_val] * len(error_grid),
            error_grid,
            density_total,
            color=color,
            linewidth=0.8,
            alpha=0.7,
            zorder=5
        )

    # Настройка осей
    ax.set_xlim(meta.get('minTime', df['time'].min()), meta.get('maxTime', df['time'].max()))
    ax.set_ylim(error_max, error_min)  # инвертируем, чтобы меньшие ошибки были ближе
    ax.set_zlim(0, df['density'].max() * 1.15)

    ax.set_xlabel('Время (с)', fontsize=LABEL_FS, labelpad=12)
    ax.set_ylabel('Ошибка (м)', fontsize=LABEL_FS, labelpad=12)
    ax.set_zlabel('Плотность вероятности', fontsize=LABEL_FS, labelpad=12)

    title_text = f'Эволюция плотности ошибок\n(точек: {len(df):,}, срезов: {len(selected_times)})'
    ax.set_title(title_text, fontsize=TITLE_FS, pad=20)

    ax.view_init(elev=30, azim=-60)
    ax.grid(True, alpha=0.2, linestyle=':')

    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path / '3d_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(out_path / '3d_plot.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"3D график (wireframe) сохранён в {out_path}")
    return True


# ============================================================================
# УПРАВЛЕНИЕ ФАЙЛАМИ И ТОЧКА ВХОДА
# ============================================================================
def cleanup_output_dir(out_dir: str, force: bool) -> bool:
    """Удаляет существующие файлы графиков, если установлен флаг force."""
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    files_to_check = ['2d_plot.png', '2d_plot.pdf', '3d_plot.png', '3d_plot.pdf']
    cleaned = False
    for fname in files_to_check:
        fpath = out_path / fname
        if fpath.exists():
            if force:
                fpath.unlink()
                print(f"Удалён: {fpath}")
                cleaned = True
            else:
                print(f"Существует (используйте --force для перезаписи): {fpath}")
    return cleaned

# ============================================================================
# МЕТРИКИ
# ============================================================================
def plot_metrics(metrics_csv_path: str, out_dir: str, split_time: Optional[float] = None, unified_csv_path: Optional[str] = None) -> bool:
    df = load_csv(pathlib.Path(metrics_csv_path))
    if df is None:
        print(f"Не найден или пуст metrics.csv: {metrics_csv_path}")
        return False

    required = ['time', 'silhouette', 'davies_bouldin']
    if not all(c in df.columns for c in required):
        print(f"metrics.csv не содержит необходимых колонок: {list(df.columns)}")
        return False

    # Если split_time не передан, пробуем определить его из unified CSV
    if split_time is None and unified_csv_path:
        df_unified = load_csv(pathlib.Path(unified_csv_path))
        if df_unified is not None:
            split_time = detect_mode_transitions(df_unified)
            if split_time:
                print(f"Определено время разделения: {split_time:.0f} с")

    df = df.sort_values('time')
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.suptitle('Метрики качества кластеризации по времени', fontsize=14, fontweight='bold')

    # Silhouette
    axes[0].plot(df['time'], df['silhouette'], 'o-', color='#1f77b4', markersize=3, linewidth=1.5)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_ylabel('Silhouette')
    axes[0].grid(True, alpha=0.3)

    # Davies–Bouldin index
    axes[1].plot(df['time'], df['davies_bouldin'], 's-', color='#d62728', markersize=3, linewidth=1.5)
    axes[1].set_ylabel('Davies–Bouldin index')
    axes[1].set_xlabel('Время (с)')
    axes[1].grid(True, alpha=0.3)

    # ГРАФИК ИНЕРЦИИ
    axes[2].plot(df['time'], df['inertia'], '^-', color='#2ca02c', markersize=3, linewidth=1.5)
    axes[2].set_ylabel('Inertia')
    axes[2].set_xlabel('Время (с)')
    axes[2].grid(True, alpha=0.3)

    if split_time is not None:
        for ax in axes:
            ax.axvline(x=split_time, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='split time')
        axes[0].legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / 'metrics_plot.png', dpi=150, bbox_inches='tight')
    plt.savefig(out_path / 'metrics_plot.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Графики метрик сохранены в {out_path}")
    return True

def main() -> int:
    parser = argparse.ArgumentParser(description='Визуализация унифицированных данных GNSS')
    parser.add_argument('csv_file', help='Путь к унифицированному CSV-файлу')
    parser.add_argument('--output-dir', default='results', help='Папка для сохранения графиков')
    parser.add_argument('--cleanup', action='store_true', help='Удалить старые графики перед генерацией')
    parser.add_argument('--force', action='store_true', help='Принудительно перезаписать существующие файлы')
    parser.add_argument('--verbose', action='store_true', help='Включить подробный вывод')
    parser.add_argument('--split-time', type=float, help='Вручную задать время разделения')
    parser.add_argument('--skip-2d', action='store_true', help='Пропустить генерацию 2D графика')
    parser.add_argument('--skip-3d', action='store_true', help='Пропустить генерацию 3D графика')
    parser.add_argument('--metrics-csv', help='Path to metrics.csv (optional)')

    args = parser.parse_args()

    if args.verbose:
        print(f"Инструмент визуализации | Вход: {args.csv_file} | Выход: {args.output_dir}")

    input_path = pathlib.Path(args.csv_file)
    if not input_path.exists():
        print(f"Ошибка: файл не найден: {args.csv_file}")
        return 1
    
    if args.metrics_csv and pathlib.Path(args.metrics_csv).exists():
        plot_metrics(args.metrics_csv, args.output_dir, split_time=args.split_time, unified_csv_path=args.csv_file)

    cleanup_output_dir(args.output_dir, args.cleanup or args.force)

    success_2d = True
    success_3d = True

    if not args.skip_2d:
        success_2d = plot_2d_unified(args.csv_file, args.output_dir, args.split_time)

    if not args.skip_3d:
        success_3d = plot_3d_unified(args.csv_file, args.output_dir)

    output_abs = pathlib.Path(args.output_dir).absolute()
    if success_2d and success_3d:
        print(f"Успешно | Результаты: {output_abs}")
        return 0
    elif success_2d or success_3d:
        print(f"Частичный успех | Результаты: {output_abs}")
        return 0
    else:
        print("Не удалось сгенерировать графики")
        return 1


if __name__ == "__main__":
    sys.exit(main())