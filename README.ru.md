# Библиотека расчёта солнечного давления — GPU & RT-Cores

[English](README.md) · [Русский](README.ru.md)

Библиотека вычисления силы и момента солнечного давления (СД) на
произвольной триангулированной геометрии космического аппарата.
Шесть равноправных алгоритмов через единственный публичный
заголовок [`SRPLibrary.h`](SM2D/SRPLibrary.h).

---

## Содержание

* [Зачем эта библиотека](#зачем-эта-библиотека)
* [Физика](#физика)
* [Шесть методов](#шесть-методов)
* [Расчёт vs визуализация](#расчёт-vs-визуализация)
* [Быстрый старт](#быстрый-старт)
* [Публичный API](#публичный-api)
* [Шарниры и артикуляция](#шарниры-и-артикуляция)
* [Визуализация](#визуализация)
* [Определение оборудования](#определение-оборудования)
* [Настройка с нуля](#настройка-с-нуля)
* [Требования и сборка](#требования-и-сборка)
* [Типичные ошибки](#типичные-ошибки)

---

## Зачем эта библиотека

Солнечное давление — неконсервативное возмущение, доминирующее в эволюции
орбит спутников с большим отношением площади к массе (солнечные паруса,
телескопы, КА с раскрытыми панелями). Для корректного учёта нужны:

* **самозатенение** деталей геометрии;
* **многократные зеркальные переотражения** между поверхностями;
* **точные оптические свойства** для каждой грани (поглощательная способность α,
  доля зеркальности μ, излучательная способность ε).

Библиотека делает это для триангуляции из HDF5, поддерживает шарниры
(вращательные и поступательные), и предоставляет 6 равнозначных алгоритмов
расчёта — от чистого CPU до OptiX на RT Cores.

---

## Физика

На каждый освещённый треугольник в направлении на Солнце ŝ силы складываются
по модели Кеннелли (Kenneally 2020):

$$
\mathbf{F}_i \;=\; \Phi_0 \cdot A_i \cdot \cos\theta_i \cdot
                   \bigl( s_{\text{coeff}}\,\hat{\mathbf{s}} \;+\;
                          n_{\text{coeff}}\,\hat{\mathbf{n}}_i \bigr)
$$

где

* $\Phi_0 = 4.56 \cdot 10^{-6}\ \text{Н/м}^2$ — поток солнечного давления на 1 а. е.
  (глобальная переменная `g_srp_phi0`, доступна для переопределения);
* $A_i$, $\hat{\mathbf{n}}_i$, $\cos\theta_i = \hat{\mathbf{n}}_i \cdot \hat{\mathbf{s}}$ —
  площадь, нормаль и косинус угла падения треугольника $i$;
* $s_{\text{coeff}} = -\bigl((1-\alpha) + \alpha(1-\mu)\bigr)$,
  $n_{\text{coeff}} = -\bigl(2\alpha\mu\cos\theta_i + \tfrac{2}{3}\alpha(1-\mu)\bigr)$;
* $\alpha = $ `reflectance`, $\mu = $ `specularity`.

Суммарный момент — $\mathbf{M} = \sum_i \mathbf{r}_i \times \mathbf{F}_i$,
где $\mathbf{r}_i$ — центроид (для центроидных методов) или точка попадания
луча (для пиксельных методов).

При наличии переотражений (`max_reflections > 0`) интенсивность каждого
последующего отскока умножается на $\alpha\mu$ зеркала.

---

## Шесть методов

Два семейства × три бэкенда:

|                | **CPU**          | **CUDA cores**   | **OptiX RT Cores** |
| -------------- | ---------------- | ---------------- | ------------------ |
| **Centroid**   | `CentroidCPU`    | `CentroidGPU`    | `CentroidRTX`      |
| **PixelGrid**  | `PixelGridCPU`   | `PixelGridGPU`   | `PixelGridRTX`     |

* **Centroid** (центроидный) — один теневой луч на треугольник. Сложность
  $O(N \log N)$ на BVH. Точность ограничена размером треугольника:
  частично освещённая грань будет либо целиком освещена, либо целиком в тени.
* **PixelGrid** (пиксельный) — параллельные лучи на двумерной сетке
  перпендикулярно направлению на Солнце с шагом `grid_step`. Сложность
  $O(1/\text{step}^2 \cdot \log N)$. Корректно разрешает скользящие углы
  и грани, меньшие сетки.

Внутри каждого метода — одинаковая физика. Различие — в способе семплирования
видимости.

Рекомендуемые значения `grid_step` (для PixelGrid):

| Размер геометрии | `grid_step` | Лучей на $5 \times 5\ \text{м}^2$ панель |
| ---------------- | ----------- | ------------------------------------ |
| Микро-спутник (CubeSat 10–30 см) | 0.005 м | ≈10⁶ |
| Малый КА (1–3 м) | 0.02 м | ≈6·10⁴ |
| **Стандарт (5–10 м)** | **0.05 м** | **≈10⁴** |
| Крупный КА (15–30 м) | 0.1 м | ≈2.5·10³ |

---

## Расчёт vs визуализация

В точности повторяется архитектура CC12: **для каждого GPU/RTX метода две
независимые реализации kernel'ов**.

### `compute(method)` — быстрый путь

Использует *bench*-варианты kernel'ов (файлы `*_bench.cu` /
`optix_shaders_*_bench.cu`). Эти варианты:

* не выделяют GPU-буферы под `bounce_levels`, `incident_dirs`, `origin_pts`;
* не выполняют `atomicMin` / `atomicCAS` для трекинга отскоков;
* не вызывают `set_bounce_globals()` после возврата.

Возвращают только `labels`, `total_force`, `total_moment`. На больших
моделях (миллионы треугольников) — экономия памяти GPU до сотен мегабайт.

> **Внимание:** `compute()` в bench-режиме возвращает **пустой** вектор
> `labels`. Для доступа к меткам освещённости используйте `computeViz()`.

### `computeViz(method)` — путь под визуализацию

Использует обычные kernel'ы (без `_bench`). Они:

* выделяют GPU-буферы под per-triangle трекинг отскоков;
* записывают в них через атомарные операции `atomicMin` / `atomicCAS`;
* копируют данные обратно на CPU и вызывают `set_bounce_globals()`.

После `computeViz()` валидны:

```cpp
const auto& bounce  = engine.getBounceLevels();   // -1=miss, 0=direct, 1+=bounce
const auto& incid   = engine.getIncidentDirs();   // направление прихода света
const auto& origins = engine.getOriginPts();      // точка вылета отражённого луча (NaN для прямого)
```

Для CPU-методов (`CentroidCPU`, `PixelGridCPU`) обе функции вызывают один
и тот же код — на CPU оверхед на bounce-трекинг пренебрежимо мал.

---

## Быстрый старт

```cpp
#include "SRPLibrary.h"
#include <cstdio>

int main()
{
    // 1. Загрузка геометрии: сканирует папку, читает первый .h5 файл
    SRPEngine engine("data3d_hdf5_0.5");

    // 2. Параметры
    engine.setSunDirection(1.0, 0.0, 0.0);   // Солнце вдоль +X (body frame)
    engine.setMaxReflections(2);             // 2 отскока
    engine.setGridStep(0.05);                // 5 см (только для PixelGrid)

    // 3. Шарниры (например — повернуть левую панель на 45°)
    engine.dataset().articulate("SolarPanel_L", JointConfig::rotY(M_PI / 4));

    // 4. Быстрый расчёт
    SRPResult r = engine.compute(SRPMethod::CentroidRTX);
    std::printf("F = [%g, %g, %g] N\n",
                r.total_force[0], r.total_force[1], r.total_force[2]);

    // 5. Тот же расчёт + данные для визуализации
    engine.computeViz(SRPMethod::PixelGridRTX);
    engine.visualizeLastResult(/*every=*/1, /*show_normals=*/false);
}
```

Распространяемый заголовок — **только** `SRPLibrary.h`. Никаких других
public-инклюдов пользователю не нужно.

> **Важно:** собирайте и запускайте только в конфигурации **Release | x64**.
> Debug CUDA-сборка использует значительно больше регистров GPU и завершается
> ошибкой `CUDA error 701: too many resources requested for launch`.

---

## Публичный API

### Типы

```cpp
struct Triangle {
    std::string ID;
    double v1_x, v1_y, v1_z;            // вершины (body frame, м)
    double v2_x, v2_y, v2_z;
    double v3_x, v3_y, v3_z;
    double normal_x, normal_y, normal_z;
    std::string component_type;          // HDF5 атрибут
    std::string component_name;          // например "S1_Mirror"
    double label = 1.0;                  // 1 — освещён, 0 — в тени

    double reflectance = 0.0;   // α ∈ [0,1] — поглощательная способность
    double specularity = 0.0;   // μ ∈ [0,1] — доля зеркального отражения
    double emissivity  = 0.0;   // ε ∈ [0,1] — излучательная способность

    double area = 0.0;
    double centroid_x = 0.0, centroid_y = 0.0, centroid_z = 0.0;
};

struct SRPResult {
    std::vector<int>      labels;        // 1 = освещён, 0 = в тени (пусто при compute())
    std::array<double, 3> total_force;   // [Fx, Fy, Fz] в Ньютонах
    std::array<double, 3> total_moment;  // [Mx, My, Mz] в Н·м
};

enum class SRPMethod {
    CentroidCPU, CentroidGPU, CentroidRTX,
    PixelGridCPU, PixelGridGPU, PixelGridRTX
};
```

### Класс `SRPEngine`

```cpp
SRPEngine(const std::string& folder_path);   // сканирует папку и загружает первый .h5

void   setSunDirection(double x, double y, double z);   // нормализуется автоматически
void   setMaxReflections(int bounces);                  // по умолчанию 0
void   setGridStep(double step);                        // по умолчанию 0.05 м
double getGridStep() const;

SRPResult compute   (SRPMethod m = SRPMethod::CentroidRTX);  // быстрый путь (labels пустые)
SRPResult computeViz(SRPMethod m = SRPMethod::CentroidRTX);  // + bounce-трекинг

const std::vector<int>&                          getLabels()       const;
const std::vector<int>&                          getBounceLevels() const;
const std::vector<std::array<double,3>>&         getIncidentDirs() const;
const std::vector<std::array<double,3>>&         getOriginPts()    const;

SatelliteDataset&       dataset();
const SatelliteDataset& dataset() const;

void visualizeLastResult(int every = 1, bool show_normals = false) const;
```

### Глобальная константа `g_srp_phi0`

```cpp
extern double g_srp_phi0;   // 4.56e-6 Н/м² на 1 а.е. — можно переопределить
```

Например, для перехода на нормированный выход (результат — в единицах
$\Phi_0 \cdot \text{м}^2$):

```cpp
g_srp_phi0 = 1.0;
engine.compute(SRPMethod::CentroidRTX);
```

---

## Шарниры и артикуляция

### Вращательные шарниры (revolute)

```cpp
auto& ds = engine.dataset();

// По стандартным осям:
ds.setJoint("Panel_L", JointConfig::rotX(M_PI / 6));      // 30° вокруг X
ds.setJoint("Panel_R", JointConfig::rotY(M_PI / 4));      // 45° вокруг Y
ds.setJoint("Boom",    JointConfig::rotZ(angle));

// Произвольная ось:
ds.setJoint("Mirror", JointConfig::rot(0.5, 0.7, 0.5, M_PI / 8));

// Точка вращения:
ds.setJoint("Hinge", JointConfig::rotZ(angle).atOrigin());        // вокруг (0,0,0)
ds.setJoint("Hinge", JointConfig::rotZ(angle).atPivot(1, 0, 0));  // вокруг (1,0,0)
// (по умолчанию — вокруг геометрического центра компонента)

ds.reload();   // применить все изменения

// Или одной строкой:
ds.articulate("Panel_L", JointConfig::rotY(M_PI / 4));   // setJoint + reload
```

### Поступательные шарниры (prismatic)

```cpp
ds.setJoint("Boom",     JointConfig::translateZ(0.5));    // 50 см вдоль Z
ds.setJoint("Antenna",  JointConfig::translate(1, 0, 0, 0.3));
ds.reload();
```

### Управление шарнирами

```cpp
ds.hasJoint("Panel_L");           // bool
ds.getJoint("Panel_L");           // JointConfig
ds.joints();                      // std::map<name, JointConfig>
ds.clearJoint("Panel_L");
ds.clearAllJoints();
ds.printJoints();                 // в stdout
```

### Загрузка разных файлов

```cpp
ds.fileNames();                   // список .h5 в папке
ds.fileCount();
ds.load(0);                       // по индексу
ds.load("custom/path/sat.h5");    // по абсолютному пути
ds.reload();                      // перечитать текущий + применить шарниры
ds.currentFile();
```

### Инспекция геометрии

```cpp
ds.triangles();                   // vector<Triangle> — вся сетка
ds.triangleCount();
ds.componentNames();              // список имён компонент
ds.componentInfos();              // name + type + count + total_area для каждой
ds.componentTriangles("Panel_L"); // указатели на треугольники компоненты
ds.hasComponent("Antenna");
ds.printInfo();
ds.printComponents();
```

---

## Визуализация

`SRPEngine::visualizeLastResult(every, show_normals)` создаёт CSV в
`results_3d/tmp_srp_result.csv` и вызывает Python-скрипт
[`visualize3d.py`](SM2D/visualize3d.py). Он строит интерактивный 3-D
рендер через Plotly (с fallback на Matplotlib, если Plotly не установлен).

Цветовая схема:

| Цвет             | Значение                              |
| ---------------- | ------------------------------------- |
| Жёлтый (#FFD700) | Прямое освещение от Солнца (bounce=0) |
| Тёмно-оранжевый  | 1-й отскок зеркального отражения      |
| Оранжево-красный | 2-й отскок                            |
| Орхидея          | 3+ отскоков                           |
| Серый            | В тени (label=0)                      |

Стрелки силы и момента — от начала координат (длина пропорциональна модулю).
Параметр `every` — прореживание для больших моделей (`every=10` рисует
каждый 10-й треугольник).

CSV-формат при `computeViz`:

```
Triangle ID, Component Type,
V1_X, V1_Y, V1_Z, V2_X, V2_Y, V2_Z, V3_X, V3_Y, V3_Z,
Normal_X, Normal_Y, Normal_Z,
Area, Reflectance, Specularity, Emissivity,
Label, Correct, ReflectionBounce,
Incident_X, Incident_Y, Incident_Z,
Origin_X, Origin_Y, Origin_Z
```

При обычном `compute` колонки `ReflectionBounce`, `Incident_*`, `Origin_*`
отсутствуют.

---

## Определение оборудования

```cpp
#include "SM3D_hw_info.cpp"   // прототипы декларируются в этой TU
```

```cpp
print_hardware_info();        // печатает таблицу CUDA-устройств и их возможностей
bool has_cuda = is_cuda_device_available();
bool has_rtx  = is_rtx_device_available();
std::string why = rtx_unavailable_reason();   // "" если RTX доступен
```

Логика выбора метода полностью на стороне пользователя — авто-fallback
отсутствует. Рекомендуемый порядок:

```cpp
SRPMethod best =
    is_rtx_device_available()  ? SRPMethod::CentroidRTX  :
    is_cuda_device_available() ? SRPMethod::CentroidGPU  :
                                 SRPMethod::CentroidCPU;
engine.compute(best);
```

---

## Настройка с нуля

Этот раздел для тех, кто разворачивает проект на новой машине.
Последовательность шагов обязательна: каждый шаг зависит от предыдущего.

> **Краткий путь:** если у вас уже установлены CUDA, OptiX и Python —
> просто запустите `.\setup.ps1` из корня репозитория в PowerShell.
> Скрипт найдёт всё сам и создаст `LocalPaths.props`.

---

### Шаг 1 — CUDA Toolkit

**Зачем:** компилятор `nvcc`, библиотеки `cuda.lib`/`cudart.lib`/`nvml.lib`, переменная среды `%CUDA_PATH%`.

Поддерживаются версии **12.4 и новее** (13.x, 14.x — всё ок). `setup.ps1`
автоматически пропишет нужную версию в `SM2D.vcxproj`.

1. Откройте [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)
2. Выберите подходящую версию → Windows → x86_64 → exe (local)
3. Установите с настройками по умолчанию.
4. Проверка после установки (в новом cmd/PowerShell):
   ```powershell
   nvcc --version
   # Ожидается: Cuda compilation tools, release XX.Y, ...
   echo $env:CUDA_PATH
   # Ожидается: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.Y
   ```

> **Важно:** в `SM2D.vcxproj` есть строки вида `CUDA 12.4.props` /
> `CUDA 12.4.targets`. VS CUDA plugin **не поддерживает** MSBuild-макрос
> в этом месте — требует буквальный номер версии. `setup.ps1` патчит
> эти строки автоматически. При ручной настройке — исправьте вручную
> (Шаг 6 ниже).

---

### Шаг 2 — OptiX SDK 9.1.0

**Зачем:** заголовки `optix.h`, PTX-файлы компилируются шейдерами.  
**Нужен только для методов `*RTX`**; без OptiX CentroidGPU/PixelGridGPU/CPU работают.

1. Зарегистрируйтесь (бесплатно) на [https://developer.nvidia.com](https://developer.nvidia.com)
2. Скачайте: [https://developer.nvidia.com/designworks/optix/download](https://developer.nvidia.com/designworks/optix/download) → OptiX 9.1.0 → Windows
3. Запустите установщик. Путь по умолчанию:
   ```
   C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0\
   ```
4. Убедитесь что файл существует:
   ```powershell
   Test-Path "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0\include\optix.h"
   # True
   ```

Если установили в другое место — укажите путь в `LocalPaths.props` (Шаг 5).

---

### Шаг 3 — vcpkg + зависимости C++

**Зачем:** библиотеки HDF5, HighFive (C++ обёртка над HDF5), pybind11 (встроенный Python).

```powershell
# 3.1 Клонировать vcpkg (если ещё не установлен)
git clone https://github.com/microsoft/vcpkg.git C:\Users\$env:USERNAME\vcpkg
C:\Users\$env:USERNAME\vcpkg\bootstrap-vcpkg.bat

# 3.2 Установить три нужных пакета (один раз, занимает ~5–10 мин)
C:\Users\$env:USERNAME\vcpkg\vcpkg install hdf5:x64-windows highfive:x64-windows pybind11:x64-windows

# 3.3 Проверка
Test-Path "C:\Users\$env:USERNAME\vcpkg\installed\x64-windows\include\hdf5.h"
Test-Path "C:\Users\$env:USERNAME\vcpkg\installed\x64-windows\include\highfive\highfive.hpp"
Test-Path "C:\Users\$env:USERNAME\vcpkg\installed\x64-windows\include\pybind11\embed.h"
# Все три должны вернуть True
```

> **Обязательно установить все три пакета!** Если пропустить `highfive` или
> `pybind11` — сборка упадёт с ошибкой `No such file or directory`.

Если vcpkg уже установлен в другом месте — найдите его:
```powershell
where.exe vcpkg
# или: echo $env:VCPKG_ROOT
```

---

### Шаг 4 — Python 3.10+

**Зачем:** только для `visualizeLastResult()`. Сам расчёт (force/moment) Python **не требует**.

Подойдёт любая версия Python 3.10 и выше (3.10, 3.11, 3.12, 3.13).

1. Скачайте последнюю версию: [https://www.python.org/downloads/](https://www.python.org/downloads/)  
   → Windows installer (64-bit)
2. При установке **отметьте** «Add Python to PATH»
3. После установки Python сам запустит `pip install` для визуализационных пакетов при первом вызове `visualizeLastResult()` — ничего делать не нужно.

Если хотите установить пакеты заранее:
```powershell
pip install numpy pandas plotly matplotlib
```

Узнать путь установки Python (нужен для Шага 5):
```powershell
python -c "import sys; print(sys.prefix)"
# Пример: C:\Users\YourName\AppData\Local\Programs\Python\Python312
```

---

### Шаг 5 — LocalPaths.props (пути для сборки)

Файл `LocalPaths.props` **не коммитится** (в `.gitignore`) — каждый разработчик создаёт свой.

#### Вариант A — автоматически (рекомендуется)

Запустите скрипт из корня репозитория в **PowerShell**:

```powershell
.\setup.ps1
```

Скрипт автоматически:
- находит CUDA, OptiX SDK, vcpkg, Python в стандартных путях;
- **патчит `SM2D.vcxproj`** — прописывает текущую версию CUDA в строках
  `CUDA X.Y.props` / `CUDA X.Y.targets` (VS CUDA plugin требует буквальный номер);
- определяет compute capability вашего GPU через `nvidia-smi`;
- проверяет наличие трёх vcpkg-пакетов (`hdf5`, `highfive`, `pybind11`);
- предлагает установить недостающие пакеты через vcpkg;
- устанавливает недостающие Python-пакеты (`numpy`, `pandas`, `plotly`, `matplotlib`);
- устанавливает `PYTHONHOME` в переменные среды пользователя;
- генерирует готовый `SM2D\LocalPaths.props`.

Если что-то установлено в нестандартное место — передайте путь явно:

```powershell
.\setup.ps1 -OptixRoot "D:\SDK\OptiX 9.1.0" -VcpkgRoot "D:\tools\vcpkg" -PythonRoot "C:\Python312"
```

Для принудительной перезаписи уже существующего файла добавьте `-Force`.

После выполнения скрипта — **перезапустите Visual Studio**, чтобы она
подхватила обновлённые переменные среды (`PYTHONHOME`).

#### Вариант Б — вручную

```powershell
# Скопировать шаблон
Copy-Item "SM2D\LocalPaths.props.template" "SM2D\LocalPaths.props"
```

Откройте `SM2D\LocalPaths.props` в любом текстовом редакторе и заполните:

| Поле | Как найти | Пример |
|------|-----------|--------|
| `<GpuArch>` | [таблица GPU](#архитектура-gpu) | `compute_86,sm_86` |
| `<OptixRoot>` | Шаг 2, путь установки | `C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0` |
| `<VcpkgRoot>` | `where.exe vcpkg` → убрать `\vcpkg.exe` | `C:\Users\Name\vcpkg` |
| `<PythonRoot>` | `python -c "import sys; print(sys.prefix)"` | `C:\Users\Name\AppData\Local\Programs\Python\Python312` |
| `<PythonLib>` | `dir "$PythonRoot\libs\python*.lib"` | `python312.lib` |

#### Архитектура GPU

| Серия GPU | `<GpuArch>` |
|-----------|-------------|
| GTX 10xx (Pascal) | `compute_61,sm_61` |
| RTX 20xx (Turing) | `compute_75,sm_75` |
| RTX 30xx (Ampere) | `compute_86,sm_86` |
| RTX 40xx (Ada) | `compute_89,sm_89` |
| RTX 50xx (Blackwell) | `compute_100,sm_100` |

`setup.ps1` определяет архитектуру автоматически. Не знаете свою серию?
```powershell
nvidia-smi --query-gpu=name --format=csv,noheader
```

---

### Шаг 6 — Сборка в Visual Studio 2022

1. Откройте `SM3D_GPU_CC10\SM2D.sln` в Visual Studio 2022
2. ⚠️ Выберите конфигурацию **Release | x64** — это обязательно!
3. **Build → Rebuild Solution**

> **Никогда не запускайте GPU-расчёты в Debug-конфигурации.**
> Debug CUDA оставляет значительно больше отладочной информации в регистрах
> каждого потока. На сложных kernel'ах это превышает лимит ресурсов SM и
> приводит к ошибке `CUDA error 701: too many resources requested for launch`.
> При этом простые (RTX) kernel'ы могут запускаться успешно в Debug,
> создавая иллюзию, что конфигурация правильная.

При первой сборке происходит следующее:
- `SM2D.vcxproj` проверяет пути из `LocalPaths.props`
- PreBuild компилирует 4 OptiX shader-файла в `.ptx` рядом с `.exe`
- Линковка собирает `SM3D.exe`

Если путь неверный — в панели **Error List** появится понятная ошибка.

---

### Шаг 7 — Запуск

```powershell
cd x64\Release
.\SM3D.exe ..\data3d_hdf5_0.5
```

При первом вызове `visualizeLastResult()` (пункт меню **14** или **C → visualize? Y**):
- программа проверит, установлены ли `numpy`, `pandas`, `plotly`, `matplotlib`
- если нет — автоматически запустит `pip install` и продолжит
- откроется браузер с интерактивным 3-D видом

---

## Датасет

Тестовая геометрия (триангулированные сетки КА в формате HDF5) опубликована
отдельно на Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20261561.svg)](https://doi.org/10.5281/zenodo.20261561)

Скачайте и распакуйте `.h5`-файлы в `SM2D/data3d_hdf5_0.5/`
(или передайте любой путь к папке в `SRPEngine`).

---

## Требования и сборка

> Пошаговая инструкция по установке всего с нуля — в разделе [Настройка с нуля](#настройка-с-нуля).

Быстрая справка по зависимостям:

| Зависимость | Версия | Назначение |
| ----------- | ------ | ---------- |
| Windows | 10/11 x64 | целевая платформа |
| Visual Studio | 2022 (v143) | C++20 (`stdcpplatest`) |
| CUDA Toolkit | **12.4+** | nvcc + `nvml.lib`/`cudart.lib` |
| OptiX SDK | 9.1.0 | только для `*RTX` методов |
| NVIDIA GPU | SM ≥ 7.5 (Turing+) | для RTX; SM ≥ 5.0 для CUDA |
| vcpkg | hdf5, highfive, pybind11 | x64-windows, все три обязательны |
| Python | **3.10+** | только для `visualizeLastResult()` |

```powershell
# Зависимости C++ (одноразово)
vcpkg install hdf5:x64-windows highfive:x64-windows pybind11:x64-windows

# Python-пакеты устанавливаются автоматически при первом вызове visualizeLastResult()
# или вручную:
pip install numpy pandas plotly matplotlib
```

PreBuild автоматически компилирует все 4 OptiX shader-файла
(`optix_shaders_center.cu`, `optix_shaders_center_bench.cu`,
`optix_shaders_pixel_grid.cu`, `optix_shaders_pixel_grid_bench.cu`) в PTX
рядом с `.exe`.

---

## Типичные ошибки

---

### ❌ `error MSB4019: The imported project "CUDA X.Y.props" was not found`

**Причина:** версия CUDA, установленная на машине, не совпадает с той, что
прописана в `SM2D.vcxproj`. VS CUDA plugin загружает `.props`-файл
**статически при открытии проекта** и требует буквальный номер версии —
MSBuild-макрос (`$(CudaVersion)`) в этом месте не работает.

**Решение:**
1. Запустите `.\setup.ps1` — он автоматически найдёт установленную версию
   CUDA и заменит `CUDA 12.4.props` / `CUDA 12.4.targets` в `SM2D.vcxproj`.
2. Или вручную откройте `SM2D.vcxproj` в текстовом редакторе и замените
   `12.4` на вашу версию (например `13.1`) в двух местах:
   ```xml
   <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 13.1.props" />
   <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 13.1.targets" />
   ```
3. Закройте и снова откройте VS (или перезагрузите проект).

---

### ❌ `fatal error C1083: Cannot open include file: 'highfive/highfive.hpp'`
### ❌ `fatal error C1083: Cannot open include file: 'pybind11/embed.h'`

**Причина:** установлен только пакет `hdf5:x64-windows`, но не
`highfive:x64-windows` и/или `pybind11:x64-windows`.

**Решение:**
```powershell
vcpkg install highfive:x64-windows pybind11:x64-windows
```
Или запустите `.\setup.ps1` — скрипт проверяет все три пакета и предложит
установить недостающие.

---

### ❌ `CUDA error: too many resources requested for launch (error 701)`

**Причина:** сборка или запуск в конфигурации **Debug** вместо **Release**.
Debug CUDA-компилятор выключает оптимизации регистров и включает полную
отладочную инструментацию — количество регистров на поток резко возрастает
и превышает лимит SM. Важный нюанс: **некоторые простые kernel'ы (например
OptiX) могут работать в Debug**, создавая иллюзию что всё нормально. Ошибка
проявляется только на сложных kernel'ах с многими переменными.

**Решение:** в Visual Studio переключите конфигурацию на **Release | x64**
и пересоберите (`Rebuild Solution`).

---

### ❌ `Visualisation skipped: Failed to import encodings module`
### ❌ `ModuleNotFoundError: No module named 'encodings'`

**Причина:** встроенный Python-интерпретатор (pybind11) не нашёл стандартную
библиотеку. Это происходит когда `PYTHONHOME` не задана или указывает на
другую версию Python.

**Решение:**

1. Запустите `.\setup.ps1` — он найдёт Python и установит `PYTHONHOME`
   в переменных среды пользователя.
2. **Обязательно перезапустите Visual Studio** после этого — VS наследует
   переменные среды только при своём запуске.
3. Или задайте вручную в PowerShell:
   ```powershell
   # Замените путь на ваш Python
   [System.Environment]::SetEnvironmentVariable(
       "PYTHONHOME",
       "C:\Users\YourName\AppData\Local\Programs\Python\Python312",
       "User"
   )
   # После этого перезапустите VS
   ```
4. Альтернатива: задать в настройках VS для этого проекта:
   `Project → Properties → Debugging → Environment → PYTHONHOME=C:\...\Python312`

---

### ❌ `The imported project "CUDA 12.4.props"` — хотя setup.ps1 сообщил `[OK] already has CUDA 13.1`

**Причина:** `setup.ps1` patched строку `<Import>`, но в файле был комментарий
с упоминанием `CUDA 13.1.props` раньше, и регулярное выражение попало в него,
а не в настоящую строку импорта.

**Диагностика:** откройте `SM2D.vcxproj` в текстовом редакторе и найдите
оба тега `<Import>` с `BuildCustomizations`. Убедитесь что в них стоит
нужная версия:
```xml
<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 13.1.props" />
...
<Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 13.1.targets" />
```
Если версия неверная — запустите `.\setup.ps1 -Force`.

---

### ❌ `RTX/OptiX not available` или пустой результат от RTX-методов

**Причина:** файлы `.ptx` не находятся рядом с `SM3D.exe`. PreBuild события
компилируют их, но это могло не произойти при частичной сборке.

**Решение:** сделайте **Rebuild Solution** (не просто Build). Проверьте что
в `x64\Release\` присутствуют:
```
optix_shaders_center.ptx
optix_shaders_center_bench.ptx
optix_shaders_pixel_grid.ptx
optix_shaders_pixel_grid_bench.ptx
```

---

### ❌ `Sun direction vector is zero`

Нулевой вектор передан в `setSunDirection`. Нормировка выполняется
автоматически, но нулевой вектор ненормируем.

**Решение:** передайте любое ненулевое направление:
```cpp
engine.setSunDirection(0.0, 1.0, 0.0);
```

---

### ❌ `No HDF5 files found in <path>`

В указанной папке нет файлов `*.h5` / `*.hdf5`, или текущая рабочая директория
задана неверно.

**Решение:** запускайте из `x64\Release\` и передавайте относительный путь:
```powershell
cd x64\Release
.\SM3D.exe ..\data3d_hdf5_0.5
```

---

### ❌ Силы CPU и GPU методов заметно различаются

**Причина (нормальная):** `Centroid*` и `PixelGrid*` по-разному семплируют
видимость — это ожидаемое различие (~1–5% на сложных геометриях).

**Причина (баг):** если `CentroidCPU` и `CentroidGPU` дают разные результаты
(> 0.1%) — вероятно нарушена логика проверки теней в одном из kernel'ов.
Убедитесь что код kernel'ов соответствует эталону (CC12).

---

---

## Лицензия

MIT License — см. файл [LICENSE](LICENSE).

---

## Контакт

**Автор:** Запевалин П.Р.  
**Email:** pav9981@yandex.ru  
**Вопросы и баги:** открывайте issue в репозитории.

---

## Цитирование

Если вы используете этот код в научной работе, пожалуйста, укажите ссылку:

> П.Р. Запевалин, «Сравнительный анализ алгоритмов самозатенения космических аппаратов»,
> *Advances in Space Research*, 2026. *(в печати)*

```bibtex
@article{zapevalin2026srp,
  title   = {Comparative analysis of spacecraft self-shadowing algorithms},
  author  = {Zapevalin, P.R.},
  journal = {Advances in Space Research},
  year    = {2026},
  note    = {in press}
}
```

---

### ❌ Программа запускается из Debug в VS (F5) вместо Release

VS запоминает конфигурацию в `.suo`-файле. Даже если переключить в Release
в toolbar, нажатие F5 может запустить Debug-бинарник.

**Решение:** **Build → Rebuild Solution** после переключения на Release,
затем снова F5. Или запускайте напрямую из `x64\Release\SM3D.exe`.
