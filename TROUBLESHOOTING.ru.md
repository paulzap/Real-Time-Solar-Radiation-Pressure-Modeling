# Типичные ошибки

[English](TROUBLESHOOTING.md) · [Русский](TROUBLESHOOTING.ru.md)

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

### ❌ `error MSB4019` — хотя `setup.ps1` сообщил `[OK] already has CUDA 13.1`

**Причина:** `setup.ps1` попал регулярным выражением в комментарий с упоминанием
`CUDA 13.1`, а не в настоящую строку `<Import>`.

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

### ❌ `fatal error C1083: Cannot open include file: 'highfive/highfive.hpp'`
### ❌ `fatal error C1083: Cannot open include file: 'pybind11/embed.h'`

**Причина:** установлен только пакет `hdf5:x64-windows`, но не
`highfive:x64-windows` и/или `pybind11:x64-windows`. Все три обязательны.

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

### ❌ `CUDA error: PTX JIT compilation failed` / пустой вывод RTX-методов

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
Убедитесь что код kernel'ов соответствует эталону.

---

### ❌ Программа запускается из Debug в VS (F5) вместо Release

VS запоминает конфигурацию в `.suo`-файле. Даже если переключить в Release
в toolbar, нажатие F5 может запустить Debug-бинарник.

**Решение:** **Build → Rebuild Solution** после переключения на Release,
затем снова F5. Или запускайте напрямую из `x64\Release\SM3D.exe`.
