import subprocess
import sys
import os

def install_requirements(requirements_path="./requirements.txt"):
    if os.path.exists(requirements_path):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
    else:
        print(f"'{requirements_path}' 파일이 존재하지 않습니다.")

# 사용 예시
install_requirements()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import glob
#메타데이터 정리 함수
def get_metadata(path):
    filename = os.path.basename(path)
    header_lines = []
    try:
        with open(path, 'r', encoding='latin1') as f:
            for line in f:
                header_lines.append(line)
                if 'End_of_Header' in line:
                    break
        print(f"파일명: {filename}")
        print("메타데이터(헤더):")
        print(''.join(header_lines))
        print('-' * 40)  # 구분선
    except Exception as e:
        print(f"{filename} 처리 중 에러 발생: {e}")

#자료 저장용 파일명 스플릿
def parse_filename(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split('_')
    if len(parts) >= 2:
        mass = parts[0].replace('k', '')
        ratio = parts[1]
        return (mass, ratio)
    return (None, None)

#data 메인 추출 함수
def extract_main_data(path) :
    f = open(path, 'r', encoding='latin1')
    line = f.readlines()
    data = line[25:]
    data = np.array([row.split('\t') for row in data], dtype=float)
    columns ="Load(N)/Stress(MPa)/Distance A(pixel)/Strain A/Distance B(pixel)/Strain B/Poisson's ratio".split('/')
    time_array = data[:, 0]
    load_array = data[:, 1]
    stress_array = data[:, 2]
    distance_a = data[:, 3]
    strain_a = data[:, 4]
    distance_b = data[:, 5]
    strain_b = data[:, 6]
    poisson_ratio = data[:, 7]

    df = pd.DataFrame({
        'x(time)' : time_array,
        "force" : load_array,
        "aa" : stress_array,
        "strain" : distance_a,
        'bb' : strain_a,
        columns[4] : distance_b,
        columns[5]: strain_b,
        columns[6]: poisson_ratio
    })
    return df


#stress (응력 계산)
def calc_stress(forces):
    # 단면적 (m^2)
    A = 1e-10
    # 예시 force 배열 (단위: N)
    force_array = forces

    # stress 계산
    stress_array = force_array / A

    # 출력
    print("Force (N):", force_array)
    print("Stress (Pa):", stress_array)
    return stress_array

#plt 그리고 저장 및 데이터 csv 저장 함수
def stress_strain_plt(stresses, strains, youngs, tensile, filename=None):
    # 그래프 그리기
    plt.figure(figsize=(8, 5))
    plt.plot(strains, stresses, 'b-', label='Stress-Strain Curve')

    plt.title('Stress vs Strain')
    plt.ylabel('Stress (MPa)')
    plt.xlabel('Strain')
    plt.grid(True)
    plt.legend()

    # 결과 텍스트 추가
    info_text = f"Young's Modulus: {youngs:.2f} Pa\nTensile Strength: {tensile:.2f} Pa"
    plt.annotate(info_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))

    #파일 저장 로직
    if filename:
        base_name = os.path.splitext(os.path.basename(filename))[0]
        result_dir = './result'
        os.makedirs(result_dir, exist_ok=True)

        save_path_img = os.path.join(result_dir, f"{base_name}_stress_strain.png")
        plt.savefig(save_path_img, dpi=300)
        print(f"그래프 저장 완료: {save_path_img}")

        # 데이터 저장 (strain, stress, youngs, tensile 상수 포함)
        df = pd.DataFrame({
            'Strain': strains,
            'Stress': stresses
        })
        # youngs, tensile 값은 모든 행에 같은 값으로 넣음
        df['YoungsModulus'] = youngs
        df['TensileStrength'] = tensile

        save_path_csv = os.path.join(result_dir, f"{base_name}_stress_strain_data.csv")
        df.to_csv(save_path_csv, index=False)
        print(f"데이터 저장 완료: {save_path_csv}")
    else:
        print("파일 이름이 주어지지 않아 저장하지 않았습니다.")

    plt.show()


#영스 모듈러스 계산
def calc_youngs(stress_array, strain_array, max_strain_threshold=0.05, apply_abs=True):
    stress_array = np.array(stress_array)
    strain_array = np.array(strain_array)


    mask = strain_array <= max_strain_threshold
    X = strain_array[mask].reshape(-1, 1)
    y = stress_array[mask]

    if len(X) < 2:
        print("선형 회귀에 충분한 점이 없습니다.")
        return np.nan

    model = LinearRegression()
    model.fit(X, y)
    modulus = model.coef_[0]

    return abs(modulus) if apply_abs else modulus

#인장강도 계산
def calc_tensile(stress_array):
    return np.max(np.array(stress_array))


#동일한 strain 값에 대해서 튀는 force값 쳐내기
def remove_duplicate_strain_outliers(df):
    grouped = df.groupby('strain')

    filtered_indices = []
    for strain, group in grouped:
        if len(group) > 1:
            #범위 [0.25 ~ 0.75] (백분율)
            q1 = group['force'].quantile(0.25)
            q3 = group['force'].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            filtered = group[(group['force'] >= lower) & (group['force'] <= upper)]
            filtered_indices.extend(filtered.index.tolist())
        else:
            filtered_indices.extend(group.index.tolist())

    return df.loc[filtered_indices].reset_index(drop=True)


#선형 회귀 -> stress-strain 이상값 제거
def detect_stress_outliers(df):
    strains = df['strain'].values.reshape(-1, 1)
    stresses = np.array(calc_stress(df['force'])).reshape(-1, 1)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_scaled = scaler_x.fit_transform(strains)
    y_scaled = scaler_y.fit_transform(stresses)

    model = LinearRegression()
    model.fit(x_scaled, y_scaled)
    y_pred_scaled = model.predict(x_scaled)

    errors = np.abs(y_scaled - y_pred_scaled).flatten()
    threshold = np.mean(errors) + 0.5 * np.std(errors)
    outlier_indices = np.where(errors > threshold)[0]

    #복원
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    plt.figure(figsize=(10, 5))
    plt.scatter(strains, stresses, label='Original Data', s=10)
    plt.scatter(strains[outlier_indices], stresses[outlier_indices], color='red', label='Detected Outliers')
    plt.plot(strains, y_pred, color='green', linewidth=2, label='Linear Regression Line')
    plt.title("Stress vs Strain with Outlier Detection")
    plt.xlabel("Strain")
    plt.ylabel("Stress")
    plt.grid(True)
    plt.legend()
    plt.show()

    df_cleaned = df.drop(index=outlier_indices).reset_index(drop=True)
    return df_cleaned

#Z-Score 기법
def detect_spikes_std(series, threshold=2):
    mean = series.mean()
    std = series.std()
    return (np.abs(series - mean) > threshold * std)

#Sliding-window 기법 (이동 경로에 따라)
def detect_spikes_rolling(series, window=5, threshold=3):
    rolling_mean = series.rolling(window, center=True).mean()
    rolling_std = series.rolling(window, center=True).std()
    return (np.abs(series - rolling_mean) > threshold * rolling_std)

# 변화율 (기울기 변화) 및 그라디언트 기법
def detect_spikes_gradient(series, threshold=1.5):
    diff = series.diff().abs()
    return diff > (threshold * diff.std())





results_dict = {}  # 결과 저장용
def final_pipeLine(paths):
    for i in paths:
        print(f"Processing file: {i}")
        get_metadata(i)
        df = extract_main_data(i)

        # Step 1: 기본 정제 (음수 제거)
        df = df[(df['force'] >= 0) & (df['strain'] >= 0)].reset_index(drop=True)

        # Step 2: 이상치 탐지 (회귀 + IQR)
        df = detect_stress_outliers(df)

        # Step 3: 같은 strain에서 force가 튀는 경우 제거
        df = remove_duplicate_strain_outliers(df)

        # ✅ Step 4: strain 값이 0.5 이하인 것만 남기기
        df = df[df['strain'] <= 0.5].reset_index(drop=True)

        mask_std = detect_spikes_std(df['strain'], threshold=3)
        mask_roll = detect_spikes_rolling(df['strain'], window=5, threshold=1.5)
        mask_grad = detect_spikes_gradient(df['strain'], threshold=3)
        spike_mask = mask_std & mask_roll & mask_grad
        df = df[~spike_mask].reset_index(drop=True)

        # Step 5: stress 계산
        stresses_cleaned = calc_stress(df['force'])
        strains_cleaned = df['strain']

        # Step 6: 결과 계산
        youngs = calc_youngs(stresses_cleaned, strains_cleaned)
        tensile = calc_tensile(stresses_cleaned)

        key = parse_filename(i)
        if key not in results_dict:
            results_dict[key] = {'youngs': [], 'tensile': []}
        results_dict[key]['youngs'].append(youngs)
        results_dict[key]['tensile'].append(tensile)

        # Step 7: 시각화
        stress_strain_plt(stresses_cleaned, strains_cleaned, youngs, tensile, filename=i)
    print("\n=== 그룹별 Young's Modulus 및 Tensile Strength 평균/표준편차 ===")
    for key, values in results_dict.items():
        y_arr = np.array(values['youngs'])
        t_arr = np.array(values['tensile'])
        print(f"📁 질량 {key[0]}k, 혼합비 {key[1]}")
        print(f"  - Young's Modulus 평균: {y_arr.mean():.2f}, 표준편차: {y_arr.std():.2f}")
        print(f"  - Tensile Strength 평균: {t_arr.mean():.2f}, 표준편차: {t_arr.std():.2f}")
        print("--------------------------------------------------")
    return results_dict


def save_summary(results_dict):
    summaries = []
    for key, values in results_dict.items():
        y_arr = np.array(values['youngs'])
        t_arr = np.array(values['tensile'])
        summary = {
            'Mass': key[0],
            'MixRatio': key[1],
            'YoungsMean': y_arr.mean(),
            'YoungsStd': y_arr.std(),
            'TensileMean': t_arr.mean(),
            'TensileStd': t_arr.std()
        }
        summaries.append(summary)

    # 모든 요약 데이터를 하나의 DataFrame으로 만들고 CSV로 저장
    df = pd.DataFrame(summaries)
    df.to_csv("./result/summary.csv", index=False)

    # 전체 요약 그래프 저장
    df_summary = pd.DataFrame(summaries)
    x_labels = df_summary['Mass'] + '-' + df_summary['MixRatio']
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    metrics = ['YoungsMean', 'YoungsStd', 'TensileMean', 'TensileStd']
    titles = ['Young\'s Modulus Mean', 'Young\'s Modulus Std', 'Tensile Strength Mean', 'Tensile Strength Std']

    for i, ax in enumerate(axs.flat):
        ax.bar(x_labels, df_summary[metrics[i]])
        ax.set_title(titles[i])
        ax.set_ylabel('MPa')
        ax.set_xticklabels(x_labels, rotation=45)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("./result/summary_subplot.png", dpi=300)
    plt.close()
# 실행
paths = glob.glob("./data/*.lvm")
results = final_pipeLine(paths)
save_summary(results)
