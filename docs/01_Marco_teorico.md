# Guía técnica para corrección, filtrado y escalamiento de señales sísmicas

**Enfoque:** ingeniería sísmica geotécnica y estructural aplicada.  
**Objetivo:** enseñar, con ecuaciones y criterios de control de calidad, cómo procesar acelerogramas, filtrar ruido, integrar a velocidad/desplazamiento y escalar o ajustar registros a un espectro objetivo.  
**Fecha:** 2026-05-08.

**Alcance y advertencia técnica.** Esta guía resume procedimientos usados en ingeniería sísmica. No reemplaza criterios normativos del proyecto, revisión de especialista ni documentación oficial del proveedor de datos. En registros de mala relación señal/ruido no existe una “corrección perfecta”: el procesamiento define un intervalo de frecuencias confiable y sacrifica información fuera de él.

---

## 1. Conceptos base y datos necesarios

### 1.1. Señal sísmica y variables

Un acelerograma digital se registra como una serie discreta:

$$
a_i = a(t_i), \qquad t_i=i\,\Delta t,\qquad i=0,\ldots,N-1
$$

donde:

- $a_i$: aceleración registrada. Puede estar en cuentas digitales, cm/s², m/s² o unidades de $g$.
- $\Delta t$: intervalo de muestreo.
- $f_s=1/\Delta t$: frecuencia de muestreo.
- $f_N=f_s/2$: frecuencia de Nyquist.
- $T_{\text{reg}}=(N-1)\Delta t$: duración total.
- $a_g(t)$: aceleración del terreno usada como excitación sísmica.
- $v(t)$: velocidad del terreno.
- $u(t)$: desplazamiento del terreno.

Para integrar:

$$
v(t)=v(0)+\int_0^t a(\tau)\,d\tau
$$

$$
u(t)=u(0)+\int_0^t v(\tau)\,d\tau
      =u(0)+v(0)t+\int_0^t (t-\tau)a(\tau)\,d\tau
$$

En registros de aceleración fuerte, errores pequeños de baja frecuencia producen deriva grande en velocidad y, sobre todo, en desplazamiento. Por eso la corrección de línea base y el filtrado de bajas frecuencias son críticos.

### 1.2. Información mínima requerida antes de procesar

Para procesar correctamente un acelerograma se debe recopilar:

1. **Metadatos del registro**
   - evento, estación, componente, orientación;
   - ubicación del evento y estación;
   - magnitud, mecanismo focal, distancia fuente-sitio;
   - profundidad, ruptura, directividad si aplica;
   - tipo de instrumento y respuesta instrumental;
   - unidades originales y factor de conversión.

2. **Muestreo**
   - $\Delta t$, $f_s$, $f_N$;
   - duración total;
   - existencia de memoria pre-evento y post-evento;
   - posibles gaps, saturación, clipping o cambios de escala.

3. **Contexto geotécnico**
   - condición de registro: superficie libre, afloramiento rocoso, profundidad, downhole;
   - $V_{S30}$, perfil $V_s$, densidad, amortiguamiento y no linealidad esperada;
   - si el registro se usará como input de propagación unidimensional, 2D/3D, talud, presa, licuación, túnel, cimentación, estructura, etc.;
   - si se necesita conservar desplazamiento permanente, pulso de velocidad o fling-step.

4. **Ventanas temporales**
   - pre-evento: ruido antes del arribo;
   - fase fuerte;
   - post-evento: tramo posterior donde el movimiento debería decaer.

5. **Objetivo de uso**
   - espectros elásticos;
   - análisis dinámico lineal o no lineal;
   - respuesta de sitio;
   - cálculo de deformaciones permanentes;
   - diseño normativo;
   - análisis probabilista o selección de suites.

El procesamiento no es único. Dos procesamientos distintos pueden ser razonables si ambos documentan su intervalo de confiabilidad, preservan las características relevantes y superan controles físicos.

---

## 2. Flujo recomendado de procesamiento de acelerogramas

Un flujo robusto es:

1. **Lectura y conversión de unidades.**
2. **Inspección inicial.**
3. **Corrección instrumental si corresponde.**
4. **Remoción de media pre-evento o línea base inicial.**
5. **Detección de saltos, offsets, clipping y gaps.**
6. **Corrección de línea base: constante, lineal, polinómica o por tramos.**
7. **Taper y padding.**
8. **Filtrado: high-pass, low-pass o band-pass, idealmente acausal de fase cero si el análisis no requiere causalidad.**
9. **Integración a velocidad y desplazamiento.**
10. **Cálculo de parámetros: PGA, PGV, PGD, intensidad de Arias, duración, espectros.**
11. **Verificación y documentación.**
12. **Escalamiento o ajuste espectral si el registro será modificado para un objetivo.**
13. **Nueva verificación posterior al escalamiento.**

La corrección debe registrarse en una ficha: versión del registro, unidades, $\Delta t$, filtros, orden, frecuencias de corte, taper, padding, tipo de baseline, restricciones usadas, coeficientes y verificaciones.

---

## 3. Corrección instrumental

### 3.1. Modelo general

Un instrumento no mide directamente $a_g(t)$. Mide una señal $y(t)$ relacionada con la aceleración por una función de transferencia $H(\omega)$:

$$
Y(\omega)=H(\omega)A(\omega)+N(\omega)
$$

donde $Y$ es la transformada de la señal registrada, $A$ la transformada de la aceleración real y $N$ ruido. Una corrección instrumental ideal sería:

$$
A(\omega)=\frac{Y(\omega)}{H(\omega)}
$$

Pero si $|H(\omega)|$ es pequeño en ciertas frecuencias, dividir por $H$ amplifica ruido. Por tanto, la deconvolución instrumental debe combinarse con filtrado y sólo aplicarse si la respuesta instrumental afecta el rango de interés.

### 3.2. Cuándo aplicar o no aplicar

- **Registros digitales modernos:** muchas bases ya entregan registros corregidos instrumentalmente. Revisar la documentación antes de repetir el proceso.
- **Instrumentos analógicos antiguos:** puede haber distorsión de amplitud y fase cerca de la frecuencia natural del transductor.
- **Rango de ingeniería típico:** para análisis de estructuras y suelos, las bandas de interés suelen estar por debajo de la zona donde la corrección instrumental es crítica, salvo registros antiguos o aplicaciones de alta frecuencia.
- **Riesgo:** la corrección instrumental puede amplificar ruido de alta frecuencia. Si se aplica, se debe usar un low-pass adecuado.

---

## 4. Corrección de línea base

### 4.1. Problema físico y numérico

La línea base es el nivel de aceleración cero. Un error pequeño:

$$
a_{\text{raw}}(t)=a_{\text{true}}(t)+\epsilon
$$

produce en velocidad:

$$
v_{\text{raw}}(t)=v_{\text{true}}(t)+\epsilon t
$$

y en desplazamiento:

$$
u_{\text{raw}}(t)=u_{\text{true}}(t)+\frac{1}{2}\epsilon t^2
$$

Por eso un sesgo constante casi imperceptible en aceleración puede generar desplazamientos irreales.

El modelo general de corrección es:

$$
a_c(t)=a_{\text{raw}}(t)-b(t)
$$

donde $b(t)$ es una línea base estimada. Puede ser constante, lineal, polinómica, por tramos o una combinación.

---

## 5. Corrección constante

### 5.1. Modelo

$$
b(t)=c_0
$$

$$
a_c(t)=a_{\text{raw}}(t)-c_0
$$

### 5.2. Elección de $c_0$

#### Método A: media pre-evento

Si existe una ventana de ruido antes del arribo $\mathcal{P}$:

$$
c_0=\frac{1}{n_{\mathcal P}}\sum_{i\in\mathcal P} a_i
$$

Es el método más simple y físicamente defendible cuando la línea base pre-evento representa el cero instrumental.

#### Método B: condición de velocidad final

Si se espera que la velocidad final sea cero:

$$
v_c(T)=\int_0^T [a_{\text{raw}}(t)-c_0]\,dt=0
$$

Entonces:

$$
c_0=\frac{1}{T}\int_0^T a_{\text{raw}}(t)\,dt
$$

En forma discreta con regla trapezoidal:

$$
c_0=\frac{1}{T}\sum_{i=1}^{N-1}\frac{a_i+a_{i-1}}{2}\Delta t
$$

#### Método C: mínimos cuadrados sobre ventanas quietas

Si hay ventanas pre y post-evento confiables:

$$
c_0=\arg\min_c \sum_{i\in\mathcal W} w_i(a_i-c)^2
$$

$$
c_0=\frac{\sum_{i\in\mathcal W}w_i a_i}{\sum_{i\in\mathcal W}w_i}
$$

### 5.3. Uso recomendado

La corrección constante es suficiente cuando:

- el registro tiene offset estable;
- el desplazamiento integrado no presenta curvatura fuerte;
- la velocidad post-evento oscila alrededor de cero;
- no hay evidencia de cambios de línea base durante la fase fuerte.

---

## 6. Corrección lineal

### 6.1. Modelo

Se resta una recta a la aceleración:

$$
b(t)=c_0+c_1 t
$$

o, mejor numéricamente, en tiempo normalizado:

$$
\tau=\frac{t-t_0}{T-t_0}
$$

$$
b(t)=c_0+c_1 \tau
$$

La normalización evita mala condición numérica cuando $t$ está en segundos grandes.

### 6.2. Coeficientes por mínimos cuadrados

Definir la matriz de diseño:

$$
\mathbf{D}=
\begin{bmatrix}
1 & \tau_0\\
1 & \tau_1\\
\vdots & \vdots\\
1 & \tau_{N-1}
\end{bmatrix}
$$

$$
\mathbf{a}=
\begin{bmatrix}
a_0\\a_1\\\vdots\\a_{N-1}
\end{bmatrix}
$$

Con pesos $\mathbf W$, la solución es:

$$
\mathbf c=(\mathbf D^T\mathbf W\mathbf D)^{-1}\mathbf D^T\mathbf W\mathbf a
$$

donde:

$$
\mathbf c=
\begin{bmatrix}
c_0\\c_1
\end{bmatrix}
$$

Los pesos deben excluir o reducir la fase fuerte si sólo se desea estimar ruido de línea base.

### 6.3. Coeficientes por restricciones de velocidad y desplazamiento final

Si se desea imponer:

$$
v_c(T)=v_T^{\text{obj}}
$$

$$
u_c(T)=u_T^{\text{obj}}
$$

con $a_c=a_{\text{raw}}-c_0-c_1\tau$, $\tau=t/T$, entonces:

$$
\int_0^T a_c(t)\,dt=v_T^{\text{obj}}-v_0
$$

$$
\int_0^T (T-t)a_c(t)\,dt=u_T^{\text{obj}}-u_0-v_0T
$$

Como:

$$
\int_0^T c_0\,dt=T c_0
$$

$$
\int_0^T c_1\frac{t}{T}\,dt=\frac{T}{2}c_1
$$

$$
\int_0^T (T-t)c_0\,dt=\frac{T^2}{2}c_0
$$

$$
\int_0^T (T-t)c_1\frac{t}{T}\,dt=\frac{T^2}{6}c_1
$$

se obtiene el sistema:

$$
\begin{bmatrix}
T & T/2\\
T^2/2 & T^2/6
\end{bmatrix}
\begin{bmatrix}
c_0\\c_1
\end{bmatrix}
=
\begin{bmatrix}
\int_0^T a_{\text{raw}}(t)\,dt-(v_T^{\text{obj}}-v_0)\\
\int_0^T (T-t)a_{\text{raw}}(t)\,dt-(u_T^{\text{obj}}-u_0-v_0T)
\end{bmatrix}
$$

Para registros sin desplazamiento permanente, se suele tomar $v_T^{\text{obj}}=0$ y $u_T^{\text{obj}}=0$. En registros near-fault con fling-step, imponer $u_T=0$ puede eliminar una deformación física.

---

## 7. Corrección polinómica general

### 7.1. Modelo

$$
b_m(t)=\sum_{k=0}^{m} c_k \tau^k
$$

$$
a_c(t)=a_{\text{raw}}(t)-b_m(t)
$$

donde $m$ es el orden:

- $m=0$: constante;
- $m=1$: lineal;
- $m=2$: cuadrático;
- $m=3$: cúbico;
- $m>3$: raramente recomendable salvo justificación fuerte.

### 7.2. Ensamblaje matricial

La matriz de diseño es:

$$
D_{ik}=\tau_i^k
$$

$$
\mathbf b=\mathbf D\mathbf c
$$

$$
\mathbf a_c=\mathbf a_{\text{raw}}-\mathbf D\mathbf c
$$

#### Mínimos cuadrados ponderados

$$
J(\mathbf c)=\|\mathbf W^{1/2}(\mathbf a_{\text{raw}}-\mathbf D\mathbf c)\|^2
$$

$$
\mathbf c=(\mathbf D^T\mathbf W\mathbf D)^{-1}\mathbf D^T\mathbf W\mathbf a_{\text{raw}}
$$

#### Mínimos cuadrados con restricciones físicas

Si se quieren imponer condiciones terminales:

$$
\mathbf C\mathbf c=\mathbf d
$$

el problema es:

$$
\min_{\mathbf c} \frac{1}{2}(\mathbf a-\mathbf D\mathbf c)^T\mathbf W(\mathbf a-\mathbf D\mathbf c)
$$

sujeto a:

$$
\mathbf C\mathbf c=\mathbf d
$$

Se resuelve con el sistema KKT:

$$
\begin{bmatrix}
\mathbf D^T\mathbf W\mathbf D & \mathbf C^T\\
\mathbf C & \mathbf 0
\end{bmatrix}
\begin{bmatrix}
\mathbf c\\
\boldsymbol\lambda
\end{bmatrix}
=
\begin{bmatrix}
\mathbf D^T\mathbf W\mathbf a\\
\mathbf d
\end{bmatrix}
$$

### 7.3. Restricciones integrales para un polinomio de aceleración

Con $\tau=t/T$, el aporte del término $c_k\tau^k$ a la velocidad final es:

$$
\int_0^T c_k\left(\frac{t}{T}\right)^k dt
=
\frac{T}{k+1}c_k
$$

El aporte al desplazamiento final es:

$$
\int_0^T (T-t)c_k\left(\frac{t}{T}\right)^k dt
=
\frac{T^2}{(k+1)(k+2)}c_k
$$

Por tanto, para un polinomio de orden $m$:

$$
\sum_{k=0}^{m}\frac{T}{k+1}c_k
=
\int_0^T a_{\text{raw}}(t)\,dt-(v_T^{\text{obj}}-v_0)
$$

$$
\sum_{k=0}^{m}\frac{T^2}{(k+1)(k+2)}c_k
=
\int_0^T (T-t)a_{\text{raw}}(t)\,dt-(u_T^{\text{obj}}-u_0-v_0T)
$$

Estas dos ecuaciones pueden formar parte de $\mathbf C\mathbf c=\mathbf d$.

### 7.4. Corrección ajustada en velocidad

A veces se integra primero el acelerograma y se observa deriva en velocidad. Se ajusta un polinomio $q_n(t)$ a la velocidad:

$$
v_{\text{raw}}(t)\approx q_n(t)=\sum_{k=0}^{n}\beta_k\tau^k
$$

Si se desea corregir la aceleración, se debe restar la derivada de ese polinomio:

$$
a_c(t)=a_{\text{raw}}(t)-\frac{dq_n(t)}{dt}
$$

Como:

$$
\frac{d}{dt}\tau^k=\frac{k}{T}\tau^{k-1}
$$

entonces:

$$
\frac{dq_n}{dt}=\sum_{k=1}^{n}\frac{k\beta_k}{T}\tau^{k-1}
$$

Consecuencia práctica:

- línea recta en velocidad $\Rightarrow$ corrección constante en aceleración;
- parábola en velocidad $\Rightarrow$ corrección lineal en aceleración;
- cúbica en velocidad $\Rightarrow$ corrección cuadrática en aceleración.

Esto evita el error frecuente de restar una curva ajustada a velocidad directamente sobre aceleración sin derivarla.

### 7.5. Elección del orden del polinomio

Regla práctica: usar el orden más bajo que satisface los controles físicos.

| Orden | Qué corrige | Riesgo | Uso típico |
|---:|---|---|---|
| 0 | Offset constante | no corrige drift curvo | registros digitales estables |
| 1 | Tendencia lineal en aceleración / curvatura en velocidad | puede modificar bajas frecuencias reales | drift suave |
| 2 | Curvatura en aceleración | mayor remoción de contenido de largo periodo | registros con deriva fuerte no explicada por offset |
| 3 | Curvaturas complejas | alto riesgo de sobreajuste | casos especiales, documentados |
| >3 | Variaciones complejas | puede “fabricar” desplazamientos físicamente plausibles pero falsos | evitar salvo investigación específica |

Criterios para elegir:

1. **Simplicidad:** elegir el menor orden que elimine deriva no física.
2. **Ventanas quietas:** el polinomio debe ajustarse preferentemente a tramos donde se conoce o se espera ausencia de movimiento.
3. **No borrar física:** si el registro contiene pulso de velocidad o desplazamiento permanente, no imponer cero desplazamiento final sin evidencia.
4. **Comparar espectros:** el polinomio no debe alterar significativamente el espectro dentro del rango de periodos de interés.
5. **Repetibilidad:** si pequeñas variaciones en la ventana producen ouputados muy distintos, el registro no es confiable para largo periodo.

---

## 8. Corrección por tramos y offsets de línea base

### 8.1. Modelo por tramos

Un registro puede tener cambios bruscos de línea base, especialmente durante la fase fuerte. Se puede modelar:

$$
b(t)=
\begin{cases}
b_0(t), & 0\le t<t_1\\
b_1(t), & t_1\le t<t_2\\
b_2(t), & t_2\le t\le T
\end{cases}
$$

con polinomios de bajo orden o constantes por segmento.

### 8.2. Modelo de offset escalonado

Un modelo simple es:

$$
b(t)=
\begin{cases}
0, & t<t_1\\
c_1, & t_1\le t<t_2\\
c_2, & t\ge t_2
\end{cases}
$$

Los coeficientes se obtienen imponiendo velocidad y desplazamiento final, o ajustando tramos de velocidad que deberían oscilar alrededor de cero.

El aporte a velocidad final es:

$$
c_1(t_2-t_1)+c_2(T-t_2)
$$

El aporte a desplazamiento final es:

$$
c_1\int_{t_1}^{t_2}(T-t)dt+c_2\int_{t_2}^{T}(T-t)dt
$$

$$
=
c_1\left[T(t_2-t_1)-\frac{t_2^2-t_1^2}{2}\right]
+c_2\left[\frac{(T-t_2)^2}{2}\right]
$$

Con objetivos $v_T^{obj}$ y $u_T^{obj}$, se resuelve un sistema $2\times2$.

### 8.3. Selección de tiempos $t_1,t_2$

Los tiempos $t_1,t_2$ pueden elegirse por:

- inspección de cambios bruscos en velocidad;
- inicio y fin de fase fuerte;
- umbrales de intensidad acumulada, por ejemplo 5 % y 95 % de intensidad de Arias;
- cambios detectados en media móvil de aceleración;
- restricciones instrumentales conocidas.

La subjetividad de $t_1,t_2$ es un problema. Se recomienda hacer análisis de sensibilidad variando los tiempos y reportar la dispersión en PGV, PGD y espectros de largo periodo.

---

## 9. Filtrado de ruido

### 9.1. Tipos de ruido

1. **Ruido de baja frecuencia**
   - deriva instrumental;
   - tilt;
   - conversión analógico-digital;
   - errores de línea base;
   - cambios térmicos;
   - memoria insuficiente post-evento.

2. **Ruido de alta frecuencia**
   - digitización;
   - limitación instrumental;
   - aliasing;
   - ruido eléctrico;
   - interacción local instrumento-soporte.

3. **Artefactos temporales**
   - gaps;
   - spikes;
   - clipping;
   - saturación;
   - cambios de ganancia.

### 9.2. Filtro high-pass o low-cut

Se usa para remover contenido de largo periodo contaminado. Para un Butterworth high-pass de orden $n$, la magnitud ideal puede escribirse como:

$$
|H_{HP}(f)|=\frac{1}{\sqrt{1+\left(\frac{f_c}{f}\right)^{2n}}}
$$

donde $f_c$ es la frecuencia de corte.

El periodo de corte asociado es:

$$
T_c=\frac{1}{f_c}
$$

Cuidado: si el análisis requiere periodos cercanos o mayores que $T_c$, el registro ya no es confiable en ese rango.

### 9.3. Filtro low-pass o high-cut

Se usa para remover ruido de alta frecuencia:

$$
|H_{LP}(f)|=\frac{1}{\sqrt{1+\left(\frac{f}{f_c}\right)^{2n}}}
$$

El $f_c$ del low-pass debe ser menor que $f_N$. Si hay remuestreo, aplicar anti-aliasing antes de reducir $f_s$.

### 9.4. Filtro band-pass

Combina high-pass y low-pass:

$$
H_{BP}(f)=H_{HP}(f;f_{c1})H_{LP}(f;f_{c2})
$$

con:

$$
f_{c1}<f<f_{c2}
$$

Se usa cuando hay ruido tanto de baja como de alta frecuencia.

### 9.5. Filtros causales y acausales

Un filtro causal se aplica sólo hacia adelante en el tiempo y puede introducir desplazamiento de fase. Un filtro acausal o de fase cero se aplica hacia adelante y hacia atrás:

$$
a_f=\text{filtfilt}(H,a)
$$

Ventajas del filtro acausal:

- no introduce fase;
- preserva mejor la localización temporal de picos;
- es usual en procesamiento de registros para espectros y análisis dinámico cuando no se requiere causalidad física estricta.

Precaución: al aplicar dos pasadas, la respuesta en amplitud efectiva cambia. Debe reportarse cómo se implementó el filtro.

### 9.6. Taper

Antes del filtrado se aplica una ventana de transición para evitar discontinuidades en los extremos:

$$
a_t(t)=w(t)a(t)
$$

Ejemplos:

- coseno medio;
- Tukey;
- Hanning en extremos.

El taper debe ser corto respecto a la señal útil, pero suficiente para reducir ringing. No debe eliminar arribo sísmico relevante.

### 9.7. Padding

El padding añade ceros o segmentos de transición antes y después del registro para que el filtro actúe sin contaminar el tramo útil. El uso de pads es especialmente importante en filtros acausales.

Procedimiento:

1. remover media o baseline preliminar;
2. añadir pad inicial y final;
3. aplicar taper en la unión registro-pad;
4. filtrar;
5. conservar pads para operaciones compatibles si así lo exige el procedimiento;
6. si se recorta el registro, verificar que no reaparezca deriva.

El retiro prematuro de pads puede generar incompatibilidad: la aceleración filtrada puede parecer correcta, pero la velocidad y el desplazamiento integrados pueden mostrar offset o tendencia.

### 9.8. Selección de frecuencias de corte

La frecuencia de corte debe elegirse con base en:

1. **Relación señal/ruido**
   - calcular FAS del registro y del ruido pre-evento;
   - elegir un rango donde la señal supere al ruido con margen aceptable.

2. **Espectro de Fourier**
   - detectar donde el contenido de baja frecuencia deja de seguir una tendencia física razonable;
   - identificar mesetas o incrementos artificiales.

3. **Inspección de velocidad y desplazamiento**
   - integrar después de filtros candidatos;
   - escoger el filtro más suave que elimina deriva no física.

4. **Rango de periodos de interés**
   - si el análisis requiere respuesta hasta $T_{max}$, el filtro high-pass no debe invalidar $T_{max}$;
   - como criterio conservador, no usar espectros cerca o por encima del periodo de corte sin justificar.

5. **Nyquist**
   - $f_{LP}$ debe estar claramente por debajo de $f_N$.

6. **Consistencia por componentes**
   - puede procesarse cada componente con su propio SNR;
   - si se rotarán componentes o se usará respuesta bidireccional, conviene usar filtros compatibles entre componentes horizontales.

### 9.9. Efecto del filtrado en parámetros

- PGA suele ser poco sensible a high-pass, pero sensible a low-pass si el corte es bajo.
- PGV es sensible a baja frecuencia.
- PGD es extremadamente sensible a baja frecuencia.
- Espectros de desplazamiento y periodos largos son muy sensibles.
- Espectros inelásticos pueden afectarse incluso fuera de la zona que parece intuitivamente afectada si hay distorsión de fase.
- Intensidad de Arias puede cambiar por low-pass severo, porque depende de $a^2(t)$.

---

## 10. Integración a velocidad y desplazamiento

### 10.1. Integración discreta trapezoidal

Velocidad:

$$
v_i=v_{i-1}+\frac{\Delta t}{2}(a_i+a_{i-1})
$$

Desplazamiento:

$$
u_i=u_{i-1}+\frac{\Delta t}{2}(v_i+v_{i-1})
$$

con $v_0=0$, $u_0=0$ salvo que haya condiciones iniciales conocidas.

### 10.2. Forma matricial

Definir un operador triangular $\mathbf L$ que integra con regla trapezoidal:

$$
\mathbf v=\mathbf L\mathbf a+v_0\mathbf 1
$$

$$
\mathbf u=\mathbf L\mathbf v+u_0\mathbf 1
$$

Entonces:

$$
\mathbf u=\mathbf L^2\mathbf a+v_0\mathbf t+u_0\mathbf 1
$$

Esta forma es útil para ajustar baseline imponiendo directamente:

$$
\mathbf v_N \approx 0,\qquad \mathbf u_N\approx u_T^{obj}
$$

o para mínimos cuadrados con penalización en drift:

$$
J(\mathbf c)=
\|\mathbf W_a(\mathbf a-\mathbf D\mathbf c)\|^2
+\alpha_v \|\mathbf W_v \mathbf L(\mathbf a-\mathbf D\mathbf c)\|^2
+\alpha_u \|\mathbf W_u \mathbf L^2(\mathbf a-\mathbf D\mathbf c)\|^2
$$

### 10.3. Integración en frecuencia

Teóricamente:

$$
V(\omega)=\frac{A(\omega)}{i\omega}
$$

$$
U(\omega)=\frac{A(\omega)}{-(\omega^2)}
$$

Pero para $\omega\to0$ se amplifica cualquier error. Por eso se debe eliminar o controlar el componente cercano a cero antes de integrar.

### 10.4. Verificación de velocidad

Una velocidad corregida razonable debería mostrar:

- valor cercano a cero antes del arribo;
- oscilación alrededor de cero después de la fase fuerte;
- ausencia de tendencia lineal post-evento;
- PGV coherente con magnitud, distancia y sitio;
- pulso de velocidad preservado si se trata de registro near-fault;
- compatibilidad entre componentes.

### 10.5. Verificación de desplazamiento

Un desplazamiento corregido razonable debería mostrar:

- valor inicial estable;
- ausencia de deriva parabólica;
- desplazamiento final estable;
- cero final si no se espera desplazamiento permanente;
- desplazamiento residual no cero si hay evidencia de fling-step, ruptura cercana o medición geodésica que lo respalde;
- PGD compatible con el tipo de evento;
- espectro de desplazamiento estable en el rango de periodos de interés.

---

## 11. Parámetros de movimiento sísmico

### 11.1. PGA, PGV y PGD

$$
PGA=\max_t |a(t)|
$$

$$
PGV=\max_t |v(t)|
$$

$$
PGD=\max_t |u(t)|
$$

PGA es robusto frente a drift de baja frecuencia, pero no representa por sí solo duración ni energía. PGV suele correlacionar mejor con demanda de estructuras de periodo intermedio. PGD es importante para periodos largos, aislamiento, taludes, presas, tuberías y near-fault.

### 11.2. Intensidad de Arias

La intensidad de Arias acumulada es:

$$
I_A(t)=\frac{\pi}{2g}\int_0^t a^2(\tau)\,d\tau
$$

La intensidad total es:

$$
I_A=I_A(T)
$$

En forma discreta:

$$
I_A(t_i)=\frac{\pi}{2g}\sum_{j=1}^{i}\frac{a_j^2+a_{j-1}^2}{2}\Delta t
$$

Si $a$ está en m/s² y $g=9.80665$ m/s², $I_A$ queda en m/s.

La curva normalizada de energía acumulada o curva de Husid es:

$$
H(t)=\frac{I_A(t)}{I_A(T)}
$$

### 11.3. Duración significativa

La duración significativa $D_{p_1-p_2}$ se define con la curva de Arias normalizada:

$$
H(t_{p_1})=p_1
$$

$$
H(t_{p_2})=p_2
$$

$$
D_{p_1-p_2}=t_{p_2}-t_{p_1}
$$

Usos comunes:

$$
D_{5-95}=t_{95}-t_{5}
$$

$$
D_{5-75}=t_{75}-t_{5}
$$

La duración significativa captura el intervalo donde se acumula una fracción de la energía del movimiento. Es clave en análisis no lineales, licuación, degradación cíclica y daño acumulado.

### 11.4. Duración bracketed y uniforme

Duración bracketed:

$$
D_B=t_{\text{último }|a|>a_{thr}}-t_{\text{primer }|a|>a_{thr}}
$$

Duración uniforme:

$$
D_U=\sum_i \Delta t \; \mathbf{1}(|a_i|>a_{thr})
$$

El umbral suele expresarse como porcentaje de PGA o valor absoluto, por ejemplo 0.05g.

### 11.5. Cumulative Absolute Velocity, CAV

$$
CAV=\int_0^T |a(t)|\,dt
$$

Discreto:

$$
CAV=\sum_{i=1}^{N-1}\frac{|a_i|+|a_{i-1}|}{2}\Delta t
$$

CAV es sensible a duración y amplitud.

### 11.6. RMS de aceleración

$$
a_{rms}=\sqrt{\frac{1}{T}\int_0^T a^2(t)\,dt}
$$

Relaciona amplitud media energética con duración.

---

## 12. Respuesta elástica de oscilador SDOF

### 12.1. Ecuación de movimiento

Para un oscilador lineal de un grado de libertad sometido a aceleración de base:

$$
m\ddot u+c\dot u+ku=-m a_g(t)
$$

donde $u(t)$ es el desplazamiento relativo. Definir:

$$
\omega_n=\sqrt{\frac{k}{m}}
$$

$$
T_n=\frac{2\pi}{\omega_n}
$$

$$
\xi=\frac{c}{2m\omega_n}
$$

La ecuación normalizada:

$$
\ddot u+2\xi\omega_n\dot u+\omega_n^2u=-a_g(t)
$$

### 12.2. Espectros

Para cada periodo $T_n$ y amortiguamiento $\xi$:

$$
S_d(T,\xi)=\max_t |u(t)|
$$

$$
S_v(T,\xi)=\omega_n S_d(T,\xi)
$$

$$
S_a(T,\xi)=\omega_n^2 S_d(T,\xi)
$$

$S_a$ es pseudoaceleración. La aceleración absoluta máxima es:

$$
S_{aa}(T,\xi)=\max_t |\ddot u(t)+a_g(t)|
$$

Para amortiguamiento bajo, $S_a$ y $S_{aa}$ son cercanas en muchos rangos, pero no idénticas.

### 12.3. Integración Newmark-beta

Para $\beta=1/4$, $\gamma=1/2$ se obtiene el método de aceleración promedio, incondicionalmente estable para sistemas lineales.

Ecuación incremental:

$$
\hat k = k+\frac{\gamma}{\beta\Delta t}c+\frac{1}{\beta\Delta t^2}m
$$

$$
\Delta \hat p_i=\Delta p_i+
m\left(\frac{1}{\beta\Delta t}\dot u_i+\frac{1}{2\beta}\ddot u_i\right)
+
c\left(\frac{\gamma}{\beta}\dot u_i+\Delta t\left(\frac{\gamma}{2\beta}-1\right)\ddot u_i\right)
$$

$$
\Delta u_i=\frac{\Delta \hat p_i}{\hat k}
$$

$$
\Delta \dot u_i=\frac{\gamma}{\beta\Delta t}\Delta u_i-\frac{\gamma}{\beta}\dot u_i+\Delta t\left(1-\frac{\gamma}{2\beta}\right)\ddot u_i
$$

$$
\Delta \ddot u_i=\frac{1}{\beta\Delta t^2}\Delta u_i-\frac{1}{\beta\Delta t}\dot u_i-\frac{1}{2\beta}\ddot u_i
$$

donde $p(t)=-ma_g(t)$.

---

## 13. Respuesta inelástica y ductilidad

### 13.1. Ecuación inelástica SDOF

$$
m\ddot u+c\dot u+f_s(u,z)=-m a_g(t)
$$

donde $f_s$ es la fuerza restauradora y $z$ representa variables internas de histéresis.

Para elastoplástico perfecto:

$$
f_y=k u_y
$$

$$
f_s=
\begin{cases}
ku, & |u|\le u_y\\
\operatorname{sign}(u)f_y, & \text{fluencia perfecta}
\end{cases}
$$

Para modelos bilineales:

$$
k_p=\alpha k
$$

donde $\alpha$ es la razón de endurecimiento post-fluencia.

### 13.2. Ductilidad

La demanda de ductilidad es:

$$
\mu=\frac{u_{max}}{u_y}
$$

donde:

- $u_{max}$: desplazamiento relativo máximo;
- $u_y$: desplazamiento de fluencia.

### 13.3. Factor de reducción por ductilidad

Para un periodo dado, el espectro elástico demanda:

$$
f_{el}=mS_a(T)
$$

Si el sistema fluye con resistencia $f_y$, el factor de reducción es:

$$
R_\mu=\frac{f_{el}}{f_y}
$$

o en términos de pseudoaceleración:

$$
R_\mu=\frac{S_{a,el}}{S_{a,y}}
$$

El cálculo de espectro inelástico de ductilidad constante consiste en iterar $f_y$ hasta que:

$$
\mu(f_y)=\mu_{\text{obj}}
$$

### 13.4. Algoritmo para espectro inelástico de ductilidad constante

Para cada periodo $T$:

1. Calcular respuesta elástica y $S_{a,el}$.
2. Elegir un valor inicial de $R$.
3. Definir $S_{a,y}=S_{a,el}/R$.
4. Calcular $f_y=mS_{a,y}$, $u_y=f_y/k$.
5. Integrar la ecuación inelástica.
6. Calcular $\mu=u_{max}/u_y$.
7. Ajustar $R$ con bisección o Newton hasta $\mu\approx\mu_{\text{obj}}$.
8. Guardar $S_{a,y}$, $R_\mu$, $u_{max}$, energía histerética.

---

## 14. Verificación de que la corrección de señal fue adecuada

### 14.1. Controles en aceleración

- Media pre-evento cercana a cero.
- Sin spikes anómalos.
- Sin saltos bruscos no físicos.
- PGA no cambia excesivamente salvo si se removió ruido evidente.
- La fase fuerte conserva forma y polaridad.

### 14.2. Controles en velocidad

- Velocidad pre-evento cercana a cero.
- No hay pendiente residual post-evento.
- PGV razonable.
- Pulsos near-fault preservados.
- No se introdujo ringing por filtro.
- La velocidad integrada no depende drásticamente de variaciones pequeñas de $f_c$.

### 14.3. Controles en desplazamiento

- Desplazamiento inicial estable.
- Desplazamiento final compatible con física del evento.
- Sin deriva parabólica.
- PGD estable.
- En near-fault: evaluar si existe fling-step; comparar con GPS/InSAR si está disponible.
- En análisis geotécnico: revisar si el desplazamiento permanente debe conservarse o no, según objetivo.

### 14.4. Controles espectrales

Comparar antes/después:

$$
R_S(T)=\frac{S_{a,\text{corr}}(T)}{S_{a,\text{raw}}(T)}
$$

Para el rango de periodos de interés, $R_S(T)$ no debería cambiar de manera no explicada. Si el high-pass corta a $T_c$, los periodos próximos a $T_c$ o superiores no deben usarse sin advertencia.

### 14.5. Controles de Fourier

- Revisar FAS de señal, ruido y señal filtrada.
- Confirmar que las frecuencias removidas coinciden con baja relación señal/ruido.
- Evitar filtros con roll-off excesivamente abrupto que produzcan ringing.
- Confirmar que el low-pass no corta contenido sísmico útil.

### 14.6. Sensibilidad

Procesar con varias alternativas razonables:

- $f_{HP}$ bajo, medio, alto;
- órdenes 2, 4, 6;
- baseline constante y lineal;
- ventanas de taper distintas;
- con y sin conservación de desplazamiento permanente.

Si las respuestas de interés varían demasiado, reportar incertidumbre o descartar el registro para esa aplicación.

### 14.7. Criterios de aceptación prácticos

Un procesamiento puede considerarse aceptable si:

1. Los parámetros de interés están dentro del rango de frecuencia confiable.
2. El registro corregido no exhibe deriva física imposible.
3. Velocidad y desplazamiento son coherentes.
4. El espectro no fue alterado artificialmente en el rango de diseño.
5. Los filtros y baseline están documentados.
6. El ouputado es estable frente a perturbaciones razonables.
7. La corrección no elimina rasgos físicos relevantes.

---

## 15. Escalamiento sísmico: propósito y categorías

El escalamiento sísmico modifica uno o más acelerogramas para representar un nivel de amenaza o un espectro objetivo.

### 15.1. Tipos principales

1. **Escalamiento lineal de amplitud**
   $$
   a_s(t)=\alpha a(t)
   $$
   No cambia duración ni contenido de frecuencias relativo.

2. **Selección y escalamiento de registros**
   Se eligen registros con magnitud, distancia, mecanismo, sitio y forma espectral compatibles, y se aplica $\alpha$.

3. **Ajuste espectral o spectral matching**
   Se modifica la forma de la señal para que su espectro coincida con un objetivo.

4. **Generación sintética o simulación**
   Se genera una señal compatible con un escenario o un espectro, con modelos estocásticos, deterministas o híbridos.

5. **Escalamiento basado en demanda inelástica**
   Se escala para igualar una deformación inelástica objetivo o demanda modal.

---

## 16. Espectro objetivo

### 16.1. Fuentes de espectro objetivo

- Espectro normativo de diseño.
- Espectro de peligro uniforme, UHS.
- Espectro condicional medio, CMS.
- Espectro condicional, CS, con variabilidad.
- Espectro específico de sitio.
- Espectro de respuesta de roca objetivo para análisis de respuesta de sitio.
- Espectro compatible con amenaza desagregada por magnitud, distancia y epsilon.
- Espectro objetivo de componente horizontal: RotD50, RotD100, máximo direccional, media geométrica, SRSS.

### 16.2. Compatibilidad conceptual

Un registro compatible no debe igualar sólo $S_a(T)$. Debe ser razonable en:

- magnitud;
- distancia;
- mecanismo;
- condición de sitio;
- duración;
- intensidad de Arias;
- PGV/PGD;
- pulsos near-fault si el escenario los requiere;
- contenido de Fourier;
- número de ciclos fuertes;
- polarización y orientación.

Dos registros con el mismo espectro elástico pueden producir respuestas no lineales distintas si difieren en duración o energía.

---

## 17. Escalamiento lineal de un registro

### 17.1. Propiedad lineal

Si:

$$
a_s(t)=\alpha a(t)
$$

entonces:

$$
v_s(t)=\alpha v(t)
$$

$$
u_s(t)=\alpha u(t)
$$

$$
S_{a,s}(T)=\alpha S_a(T)
$$

$$
I_{A,s}=\alpha^2 I_A
$$

$$
CAV_s=\alpha CAV
$$

La duración significativa no cambia si la escala es positiva y no hay umbral absoluto; sí puede cambiar si se usa duración bracketed con umbral absoluto.

### 17.2. Escalamiento a un periodo único

Para igualar el espectro en $T^*$:

$$
\alpha=\frac{S_{a,\text{obj}}(T^*)}{S_{a,\text{rec}}(T^*)}
$$

Uso típico:

- estructura dominada por primer modo;
- análisis preliminar;
- selección por $S_a(T_1)$.

### 17.3. Escalamiento por mínimos cuadrados lineales

Para igualar un rango de periodos $T_j$:

$$
\min_\alpha \sum_j w_j[\alpha S_j-S_j^*]^2
$$

donde:

$$
S_j=S_{a,\text{rec}}(T_j)
$$

$$
S_j^*=S_{a,\text{obj}}(T_j)
$$

La solución es:

$$
\alpha=\frac{\sum_j w_j S_jS_j^*}{\sum_j w_j S_j^2}
$$

### 17.4. Escalamiento por mínimos cuadrados en logaritmos

Como los espectros varían en órdenes de magnitud, suele preferirse:

$$
\min_{\ln\alpha}\sum_j w_j[\ln(\alpha S_j)-\ln(S_j^*)]^2
$$

$$
\ln\alpha=\frac{\sum_j w_j[\ln S_j^*-\ln S_j]}{\sum_j w_j}
$$

$$
\alpha=\exp\left(\frac{\sum_j w_j\ln(S_j^*/S_j)}{\sum_j w_j}\right)
$$

Ventaja: penaliza ratios, no diferencias absolutas.

### 17.5. Selección del rango de periodos

Para estructuras:

$$
T_{\min}=c_1T_1,\qquad T_{\max}=c_2T_1
$$

Valores frecuentes son $0.2T_1$ a $1.5T_1$ o $2T_1$, pero la norma aplicable manda.

Para geotecnia:

- respuesta de sitio: rango donde se espera amplificación del perfil;
- taludes y presas: periodos asociados a modos deformacionales;
- licuación: además del espectro, revisar duración, CSR equivalente e intensidad energética;
- cimentaciones y túneles: periodos de interacción suelo-estructura.

---

## 18. Escalamiento de una suite de registros

### 18.1. Media aritmética y geométrica

Para $n$ registros escalados:

$$
\bar S_{\text{arit}}(T_j)=\frac{1}{n}\sum_{i=1}^{n}\alpha_i S_{ij}
$$

$$
\bar S_{\text{geo}}(T_j)=
\exp\left[
\frac{1}{n}\sum_{i=1}^{n}\ln(\alpha_iS_{ij})
\right]
$$

La media geométrica es coherente con la distribución lognormal usual de ordenadas espectrales.

### 18.2. Optimización de factores individuales

Se puede resolver:

$$
\min_{\alpha_1,\ldots,\alpha_n}
\sum_j w_j
\left[
\ln\bar S(T_j)-\ln S_j^*
\right]^2
+
\eta\sum_i[\ln\alpha_i-\ln\alpha_0]^2
$$

donde $\eta$ penaliza factores extremos.

### 18.3. Restricciones

Ejemplos de restricciones:

$$
\alpha_{\min}\le\alpha_i\le\alpha_{\max}
$$

$$
\bar S(T_j)\ge \rho S_j^*
$$

donde $\rho$ puede ser 0.9, 1.0 u otro valor de acuerdo con norma o criterio de proyecto.

### 18.4. Rotación y componentes horizontales

Para dos componentes $x,y$:

- media geométrica:
  $$
  S_{gm}(T)=\sqrt{S_x(T)S_y(T)}
  $$

- SRSS:
  $$
  S_{SRSS}(T)=\sqrt{S_x^2(T)+S_y^2(T)}
  $$

- RotD50:
  mediana sobre orientaciones rotadas.

- RotD100:
  máximo sobre orientaciones.

En análisis 2D/3D debe decidirse si se escala un par con un factor común o cada componente por separado. Un factor común preserva la relación física entre componentes.

---

## 19. Selección de registros antes de escalar

### 19.1. Variables sismológicas

Seleccionar registros compatibles con:

- $M_w$;
- distancia $R_{rup}$, $R_{jb}$, $R_{hyp}$;
- mecanismo: normal, inversa, strike-slip, subducción interplaca/intraplaca;
- profundidad;
- condición de sitio;
- cuenca;
- directividad;
- hanging-wall;
- duración esperada;
- contenido de pulsos.

### 19.2. Variables de ingeniería

- forma espectral;
- amplitud compatible para evitar factores excesivos;
- PGA, PGV, PGD;
- Arias y duración;
- CAV;
- FAS;
- calidad del procesamiento;
- número de componentes disponibles;
- orientación.

### 19.3. Compatibilidad con escenario

El registro escalado debe representar el escenario de amenaza. Un registro de magnitud pequeña escalado fuertemente puede igualar $S_a(T_1)$ pero no la duración ni el contenido de baja frecuencia de un evento grande.

---

## 20. Ajuste espectral o spectrum matching

### 20.1. Objetivo

Modificar $a(t)$ para que:

$$
S_{a,m}(T_j,\xi)\approx S_{a,obj}(T_j,\xi)
$$

para periodos $T_j$ y, a veces, varios amortiguamientos $\xi$.

### 20.2. Ajuste en frecuencia

Se transforma el registro:

$$
a(t)\xrightarrow{\mathcal F} A(f)=|A(f)|e^{i\phi(f)}
$$

Se modifica la amplitud:

$$
|A_m(f)|=M(f)|A(f)|
$$

manteniendo fase:

$$
A_m(f)=|A_m(f)|e^{i\phi(f)}
$$

Luego:

$$
a_m(t)=\mathcal F^{-1}[A_m(f)]
$$

Ventajas:

- simple;
- eficiente;
- controla contenido de Fourier.

Desventajas:

- puede alterar no estacionariedad;
- puede distribuir energía en tiempos no realistas;
- puede distorsionar pulsos near-fault;
- igualar Fourier no garantiza igualar espectro de respuesta sin iteraciones.

### 20.3. Ajuste en tiempo mediante wavelets

Modelo:

$$
a_m(t)=\alpha a(t)+\sum_{k=1}^{K} c_k\psi_k(t)
$$

donde $\psi_k(t)$ son wavelets o funciones de ajuste localizadas en tiempo y frecuencia.

El error espectral puede definirse como:

$$
e_j=\ln S_{a,obj}(T_j)-\ln S_{a,m}(T_j)
$$

Linealizando:

$$
\Delta \mathbf e\approx \mathbf B\mathbf c
$$

donde:

$$
B_{jk}=\frac{\partial \ln S_a(T_j)}{\partial c_k}
$$

Se resuelve:

$$
\min_{\mathbf c}\|\mathbf W^{1/2}(\mathbf e-\mathbf B\mathbf c)\|^2+\lambda\|\mathbf c\|^2
$$

La solución amortiguada:

$$
\mathbf c=(\mathbf B^T\mathbf W\mathbf B+\lambda\mathbf I)^{-1}\mathbf B^T\mathbf W\mathbf e
$$

Luego se actualiza el registro y se itera.

### 20.4. Tiempo de aplicación de wavelets

Para cada periodo $T_j$, el ajuste suele aplicarse cerca del tiempo $t_j$ donde el oscilador de periodo $T_j$ alcanza su máxima respuesta. Esto preserva mejor la no estacionariedad del registro.

### 20.5. Ajuste simultáneo de varios amortiguamientos

Si se ajusta para varios $\xi_l$:

$$
e_{jl}=\ln S_{a,obj}(T_j,\xi_l)-\ln S_{a,m}(T_j,\xi_l)
$$

La matriz $\mathbf B$ apila filas para cada par $(T_j,\xi_l)$.

### 20.6. Control de drift en spectrum matching

Un problema clásico es que las wavelets pueden introducir velocidad o desplazamiento residual. Para evitarlo:

- usar wavelets con integral cero:
  $$
  \int_{-\infty}^{\infty}\psi(t)\,dt=0
  $$

- y, si se desea evitar desplazamiento residual:
  $$
  \int_{-\infty}^{\infty}\int_{-\infty}^{t}\psi(s)\,ds\,dt=0
  $$

- aplicar baseline posterior sólo si no distorsiona el ajuste;
- usar funciones tapered-cosine o wavelets diseñadas para no producir drift;
- verificar velocidad y desplazamiento después del ajuste.

### 20.7. Parámetros de ajuste

Se debe definir:

- rango de periodos $[T_{\min},T_{\max}]$;
- amortiguamiento;
- puntos por década en $T$;
- tolerancia;
- número máximo de iteraciones;
- regularización $\lambda$;
- límites de amplitud de wavelets;
- ventanas de tiempo permitidas;
- filtros posteriores si se permiten;
- criterio de rechazo por modificación excesiva.

### 20.8. Tolerancia

Error relativo:

$$
r_j=\frac{S_{a,m}(T_j)-S_{a,obj}(T_j)}{S_{a,obj}(T_j)}
$$

Error logarítmico:

$$
e_j=\ln\left[\frac{S_{a,m}(T_j)}{S_{a,obj}(T_j)}\right]
$$

Criterios frecuentes:

- $|r_j|\le 5\%$ para ajuste estricto;
- $|r_j|\le 10\%$ para ajuste práctico;
- promedio de suite no inferior a un porcentaje objetivo;
- tolerancias distintas para periodos críticos y secundarios.

La tolerancia debe obedecer norma o especificación de proyecto.

### 20.9. Verificación de modificación excesiva

Comparar original escalado y ajustado:

$$
\Delta a(t)=a_m(t)-\alpha a(t)
$$

Indicadores:

$$
RMS_\Delta=\sqrt{\frac{\int \Delta a^2(t)dt}{\int [\alpha a(t)]^2dt}}
$$

$$
\frac{I_{A,m}}{I_{A,\alpha}}
$$

$$
\frac{D_{5-95,m}}{D_{5-95,\alpha}}
$$

$$
\frac{PGV_m}{PGV_\alpha},\qquad \frac{PGD_m}{PGD_\alpha}
$$

También revisar visualmente:

- aceleración;
- velocidad;
- desplazamiento;
- Arias acumulada;
- espectro;
- FAS;
- energía por bandas;
- preservación de pulsos.

---

## 21. Escalamiento basado en respuesta inelástica

### 21.1. Idea

En vez de escalar a $S_a(T_1)$, se escala cada registro para que produzca una deformación inelástica objetivo en un sistema equivalente.

$$
u_{\text{inel}}(\alpha a, T,\mu, \xi, f_y)=u_{\text{obj}}
$$

Como la respuesta inelástica no escala linealmente con $\alpha$, se usa iteración.

### 21.2. Algoritmo

1. Definir sistema SDOF equivalente.
2. Definir deformación objetivo $\hat u$ o ductilidad objetivo $\mu$.
3. Elegir $\alpha_0$.
4. Integrar respuesta inelástica con $\alpha_k a(t)$.
5. Calcular error:
   $$
   e_k=u_{\max,k}-\hat u
   $$
6. Actualizar $\alpha$ por bisección, secante o Newton.
7. Repetir hasta:
   $$
   \left|\frac{u_{\max,k}-\hat u}{\hat u}\right|\le tol
   $$

### 21.3. Ventaja

Puede reducir dispersión de demandas en estructuras dominadas por un modo inelástico.

### 21.4. Limitaciones

- depende del modelo estructural equivalente;
- no necesariamente controla demanda en modos superiores;
- no es transferible a otras estructuras;
- puede esconder incompatibilidad sismológica del registro.

---

## 22. Intensidad de Arias, duración y escalamiento

### 22.1. Efecto del escalamiento lineal

Si $a_s=\alpha a$:

$$
I_{A,s}=\alpha^2 I_A
$$

$$
H_s(t)=H(t)
$$

Por tanto, $D_{5-95}$ no cambia si se define por porcentaje de Arias. Esto significa que un registro corto escalado a un espectro alto sigue siendo corto; no adquiere la duración de un terremoto grande.

### 22.2. Efecto del spectral matching

El ajuste espectral puede modificar:

- $I_A$;
- duración efectiva si redistribuye energía;
- CAV;
- número de ciclos fuertes;
- pulsos de velocidad;
- PGV/PGD.

Por eso se debe verificar energía y duración después del ajuste.

### 22.3. Relevancia para daño no lineal

En sistemas degradantes, licuación, fatiga, acumulación de deformación cortante, asentamientos y daño cíclico, dos registros espectralmente compatibles pueden producir respuestas distintas si tienen diferente duración e intensidad de Arias.

---

## 23. Procedimiento integral para corrección y filtrado

### 23.1. Paso a paso

#### Paso 1: cargar registro

- leer $a_i$;
- convertir a m/s²;
- confirmar $\Delta t$;
- detectar NaN, gaps y spikes.

#### Paso 2: inspección

- graficar $a(t)$;
- calcular PGA;
- FAS preliminar;
- integrar sin corrección sólo para diagnóstico.

#### Paso 3: definir ventanas

- pre-evento $\mathcal P$;
- fase fuerte;
- post-evento $\mathcal Q$.

#### Paso 4: remover media pre-evento

$$
a_1(t)=a_{\text{raw}}(t)-\bar a_{\mathcal P}
$$

#### Paso 5: evaluar drift

- integrar $a_1$;
- revisar $v(t)$, $u(t)$;
- si hay drift, probar baseline constante/lineal/polinómica.

#### Paso 6: elegir baseline

Empezar con:

$$
b_0=c_0
$$

Si no basta:

$$
b_1=c_0+c_1\tau
$$

Si no basta y hay justificación:

$$
b_2=c_0+c_1\tau+c_2\tau^2
$$

Usar KKT si se imponen $v_T,u_T$.

#### Paso 7: taper y pad

- taper corto en extremos;
- padding suficiente;
- documentar longitud.

#### Paso 8: elegir filtro

- high-pass por SNR y drift;
- low-pass por ruido de alta frecuencia y Nyquist;
- preferir fase cero para espectros y análisis dinámico no causal;
- evitar filtros múltiples innecesarios.

#### Paso 9: integrar

- trapezoidal o método equivalente;
- conservar unidades.

#### Paso 10: calcular parámetros

- PGA, PGV, PGD;
- $I_A$;
- $D_{5-75}$, $D_{5-95}$;
- CAV;
- espectros.

#### Paso 11: control de calidad

- comparar antes/después;
- sensibilidad;
- registrar decisiones.

### 23.2. Pseudocódigo

```text
entrada: a_raw, dt, unidades, ventanas, objetivo
a = convertir_a_m_s2(a_raw)
inspeccionar(a)

pre = definir_ventana_pre_evento(a)
post = definir_ventana_post_evento(a)

a0 = a - media(a[pre])

para cada baseline candidato:
    calcular coeficientes c
    a_b = a0 - D c
    aplicar taper y padding
    aplicar filtro candidato
    integrar a v,u
    calcular espectros y parámetros
    evaluar criterios

seleccionar el menor procesamiento que satisface criterios
guardar a_corr, v_corr, u_corr, metadatos
```

---

## 24. Procedimiento integral para escalamiento a espectro objetivo

### 24.1. Paso a paso

#### Paso 1: definir objetivo

- espectro $S_{a,obj}(T,\xi)$;
- amortiguamiento;
- componente objetivo;
- rango $[T_{\min},T_{\max}]$;
- tolerancia;
- número de registros.

#### Paso 2: seleccionar candidatos

Filtrar base de datos por:

- $M_w$;
- distancia;
- mecanismo;
- $V_{S30}$;
- calidad;
- duración;
- pulse-like o no pulse-like;
- disponibilidad de componentes.

#### Paso 3: procesar o verificar procesamiento

No escalar registros con drift no resuelto. Confirmar filtro y baseline.

#### Paso 4: calcular espectros

Para cada registro:

$$
S_{ij}=S_{a,i}(T_j)
$$

#### Paso 5: calcular factor lineal

Usar uno de:

$$
\alpha_i=\frac{S_{obj}(T^*)}{S_i(T^*)}
$$

$$
\alpha_i=\frac{\sum_jw_jS_{ij}S^*_j}{\sum_jw_jS_{ij}^2}
$$

$$
\alpha_i=\exp\left(\frac{\sum_jw_j\ln(S^*_j/S_{ij})}{\sum_jw_j}\right)
$$

#### Paso 6: evaluar suite

$$
\bar S(T_j)\approx S_{obj}(T_j)
$$

Verificar que no haya déficits críticos.

#### Paso 7: si no cumple, decidir

- cambiar registros;
- ajustar pesos;
- permitir mayor $\alpha$;
- aplicar spectral matching;
- redefinir rango;
- aumentar número de registros.

#### Paso 8: si se aplica spectral matching

- iniciar desde registro escalado;
- ajustar por wavelets o frecuencia;
- controlar drift;
- verificar energía, duración y forma temporal.

#### Paso 9: verificación final

- espectro objetivo;
- aceleración, velocidad, desplazamiento;
- Arias y duración;
- PGV/PGD;
- FAS;
- estabilidad numérica;
- documentación.

---

## 25. Condiciones especiales en ingeniería geotécnica

### 25.1. Análisis de respuesta de sitio

Debe definirse si la señal corresponde a:

- **outcrop motion:** movimiento en afloramiento rocoso;
- **within motion:** movimiento dentro del medio, a profundidad;
- **surface motion:** movimiento en superficie.

Para análisis unidimensional equivalente-lineal o no lineal, usar el tipo correcto. Un error outcrop/within puede duplicar o subestimar amplitudes.

### 25.2. Deconvolución

Si se requiere obtener el movimiento en roca basal desde un registro de superficie:

$$
A_{\text{base}}(\omega)=\frac{A_{\text{surface}}(\omega)}{H_{\text{site}}(\omega)}
$$

donde $H_{\text{site}}$ depende del perfil $V_s$, densidad, amortiguamiento y no linealidad. La deconvolución puede amplificar ruido cerca de ceros de transferencia, por lo que requiere regularización y filtros.

### 25.3. Licuación y deformación acumulada

Para licuación no basta igualar espectro:

- revisar duración;
- número de ciclos equivalentes;
- Arias;
- CAV;
- contenido de frecuencias;
- CSR inducido;
- degradación de rigidez y presión de poros.

### 25.4. Presas, taludes y desplazamientos permanentes

Si se calculan desplazamientos permanentes:

- evitar high-pass que remueva desplazamientos de largo periodo relevantes;
- verificar PGV y PGD;
- usar registros compatibles con magnitud y duración;
- revisar sensibilidad a baseline;
- no imponer desplazamiento final cero si la física del problema admite desplazamiento residual.

### 25.5. Near-fault, pulso y fling-step

Los registros cercanos a falla pueden tener:

- pulso de velocidad de directividad;
- desplazamiento permanente;
- polarización fault-normal;
- alto PGV;
- demanda elevada en periodos largos.

El filtrado high-pass y el ajuste espectral pueden destruir estos rasgos. Para estos casos:

- procesar con cuidado extremo;
- revisar velocidad y desplazamiento;
- usar datos geodésicos si están disponibles;
- no forzar espectro suave si altera el pulso;
- considerar selección específica de registros pulse-like.

---

## 26. Errores frecuentes

1. Integrar aceleración sin corregir offset.
2. Imponer desplazamiento final cero en registros con desplazamiento permanente físico.
3. Usar filtros sin reportar frecuencia, orden y fase.
4. Recortar pads después de filtrar sin verificar velocidad/desplazamiento.
5. Elegir high-pass por estética de desplazamiento, invalidando periodos de diseño.
6. Usar registros con escala excesiva sin revisar magnitud, distancia y duración.
7. Ajustar espectralmente hasta calzar el objetivo pero distorsionar la señal.
8. Comparar sólo $S_a$ y olvidar Arias, duración, PGV, FAS y desplazamiento.
9. Procesar componentes horizontales de forma incompatible y luego rotarlas.
10. Usar aceleración en $g$ dentro de fórmulas que esperan m/s².
11. Mezclar pseudoaceleración con aceleración absoluta sin distinguirlas.
12. No documentar la versión del registro ni parámetros de procesamiento.

---

## 27. Plantilla de control de calidad

### 27.1. Registro

- Evento:
- Estación:
- Componente:
- $\Delta t$:
- Unidades:
- Fuente:
- Versión:
- Registro raw/corregido original:

### 27.2. Procesamiento

- Corrección instrumental:
- Media removida:
- Baseline:
- Orden polinómico:
- Coeficientes:
- Ventanas usadas:
- Taper:
- Padding:
- Filtro high-pass:
- Filtro low-pass:
- Tipo de filtro:
- Fase:
- Software/código:

### 27.3. Parámetros ouputantes

- PGA:
- PGV:
- PGD:
- $I_A$:
- $D_{5-75}$:
- $D_{5-95}$:
- CAV:
- Rango confiable de periodos:

### 27.4. Verificaciones

- Aceleración estable:
- Velocidad final:
- Desplazamiento final:
- FAS señal/ruido:
- Espectro antes/después:
- Sensibilidad:
- Observaciones:

### 27.5. Escalamiento

- Espectro objetivo:
- Rango de periodos:
- Damping:
- Factor $\alpha$:
- Método de ajuste:
- Tolerancia:
- Error máximo:
- Error RMS:
- Verificación Arias/duración:
- Verificación velocidad/desplazamiento:
- Aceptado/rechazado:

---

## 28. Ejemplo conceptual de cálculo de baseline polinómico

Supóngase un registro de duración $T=40$ s. Luego de remover la media pre-evento, la integración da:

$$
\int_0^T a_{\text{raw}}(t)dt=0.08 \text{ m/s}
$$

$$
\int_0^T (T-t)a_{\text{raw}}(t)dt=1.60 \text{ m}
$$

Se desea $v_T=0$, $u_T=0$, $v_0=u_0=0$. Usar baseline lineal:

$$
b(t)=c_0+c_1\frac{t}{T}
$$

Sistema:

$$
\begin{bmatrix}
40 & 20\\
800 & 266.667
\end{bmatrix}
\begin{bmatrix}
c_0\\c_1
\end{bmatrix}
=
\begin{bmatrix}
0.08\\1.60
\end{bmatrix}
$$

La solución produce $c_0,c_1$, se resta $b(t)$, se integra y se verifica. Si el espectro cambia demasiado en el rango de interés, se debe revisar el modelo.

---

## 29. Ejemplo conceptual de escalamiento lineal

Registro con espectro a 5 %:

$$
S_{rec}(T_1=1.0s)=0.35g
$$

Objetivo:

$$
S_{obj}(1.0s)=0.70g
$$

Factor:

$$
\alpha=\frac{0.70}{0.35}=2.0
$$

Entonces:

$$
a_s(t)=2a(t)
$$

$$
PGV_s=2PGV
$$

$$
I_{A,s}=4I_A
$$

Después de escalar se recalcula el espectro y se verifica el rango completo, no sólo $T_1$.

---

## 30. Ejemplo conceptual de escalamiento en logaritmos

Periodos $T_j$, pesos $w_j=1$:

| $T_j$ | $S_{obj}$ | $S_{rec}$ | $\ln(S_{obj}/S_{rec})$ |
|---:|---:|---:|---:|
| 0.2 | 1.0 | 0.8 | 0.223 |
| 0.5 | 0.9 | 0.5 | 0.588 |
| 1.0 | 0.6 | 0.4 | 0.405 |

$$
\ln\alpha=\frac{0.223+0.588+0.405}{3}=0.405
$$

$$
\alpha=e^{0.405}=1.50
$$

---

## 31. Recomendaciones finales

1. **No procesar a ciegas.** Siempre inspeccionar aceleración, velocidad, desplazamiento, FAS y espectros.
2. **Preferir correcciones simples.** Si se requiere un polinomio alto para que el desplazamiento “se vea bien”, probablemente el registro no es confiable a largo periodo.
3. **El filtro define el rango usable.** No usar ordenadas espectrales más allá del rango que el filtrado permite.
4. **Conservar física relevante.** Pulsos, duración, Arias y desplazamiento residual pueden ser más importantes que una coincidencia perfecta de $S_a$.
5. **Escalar no equivale a seleccionar.** Primero se elige un registro físicamente compatible; luego se escala.
6. **Spectrum matching requiere control adicional.** Un espectro perfecto puede esconder una señal físicamente degradada.
7. **La suite importa más que un registro aislado.** Para diseño o evaluación, controlar media, dispersión y diversidad.
8. **Documentar todo.** Sin parámetros de procesamiento, el ouputado no es auditable.

---

## 32. Referencias técnicas principales

[R1] Boore, D. M., & Bommer, J. J. (2005). *Processing of strong-motion accelerograms: needs, options and consequences*. Soil Dynamics and Earthquake Engineering, 25, 93–115. DOI: 10.1016/j.soildyn.2004.10.007. Disponible en: https://www.daveboore.com/pubs_online/record_processing_sdee_final.pdf

[R2] Boore, D. M. (2005). *On Pads and Filters: Processing Strong-Motion Data*. Bulletin of the Seismological Society of America, 95(2), 745–750. DOI: 10.1785/0120040160. Disponible en USGS: https://pubs.usgs.gov/publication/70029312

[R3] NIST/NEHRP Consultants Joint Venture. (2011). *Selecting and Scaling Earthquake Ground Motions for Performing Response-History Analyses*. NIST GCR 11-917-15. Disponible en: https://www.nist.gov/publications/selecting-and-scaling-earthquake-ground-motions-performing-response-history-analyses

[R4] Kalkan, E., & Chopra, A. K. (2010). *Practical Guidelines to Select and Scale Earthquake Records for Nonlinear Response History Analysis of Structures*. USGS Open-File Report 2010-1068. Disponible en: https://pubs.usgs.gov/of/2010/1068/

[R5] PEER Ground Motion Database. Documentación técnica de búsqueda, selección y escalamiento de registros. Disponible en: https://ngawest2.berkeley.edu/site/documentation

[R6] Al Atik, L., & Abrahamson, N. (2010). *An Improved Method for Nonstationary Spectral Matching*. Earthquake Spectra, 26(3), 601–617. DOI: 10.1193/1.3459159.

[R7] Hancock, J., Watson-Lamprey, J., Abrahamson, N. A., Bommer, J. J., Markatis, A., McCoy, E., & Mendis, R. (2006). *An improved method of matching response spectra of recorded earthquake ground motion using wavelets*. Journal of Earthquake Engineering, 10(sup001), 67–89. DOI: 10.1080/13632460609350629.

[R8] SeismoSoft. (2025). *SeismoMatch Help: Spectral matching*. Disponible en: https://help.seismosoft.com/seismomatch/2025/loading_%26_preprocessing/spectral_matching.htm

[R9] SeismoSoft. (2025). *SeismoSignal Help: Ground motion parameters*. Disponible en: https://help.seismosoft.com/seismosignal/2025/spectra_%26_ground_motion_parameters/ground_motion_parameters.htm

[R10] Arias, A. (1970). *A measure of earthquake intensity*. En R. J. Hansen (Ed.), Seismic Design for Nuclear Power Plants. MIT Press.

[R11] Trifunac, M. D., & Brady, A. G. (1975). *A study on the duration of strong earthquake ground motion*. Bulletin of the Seismological Society of America, 65(3), 581–626.

[R12] Alva, R. E., Pinzón, L. A., & Pujades, L. G. (2022). *Intensidad de Arias y duración significativa en análisis dinámico de estructuras*. Ingeniería, 32(2). DOI: 10.15517/ri.v32i2.49580. Disponible en: https://www.scielo.sa.cr/scielo.php?pid=S2215-26522022000200001&script=sci_arttext

[R13] Chopra, A. K. *Dynamics of Structures: Theory and Applications to Earthquake Engineering*. Referencia general para respuesta elástica e inelástica SDOF/MDOF, Newmark y espectros.

[R14] Newmark, N. M. (1959). *A method of computation for structural dynamics*. Journal of the Engineering Mechanics Division, ASCE.

[R15] COSMOS / Center for Engineering Strong Motion Data. Manuales y documentación de registros corregidos/no corregidos y espectros. Portal: https://www.strongmotioncenter.org/

---

## 33. Lista corta de comprobación operacional

Antes de usar un registro en análisis:

- [ ] Confirmé unidades y $\Delta t$.
- [ ] Revisé si el registro ya estaba corregido.
- [ ] Removí media pre-evento si corresponde.
- [ ] Evalué línea base constante, lineal y, si fue necesario, polinómica.
- [ ] No usé polinomios altos sin justificación.
- [ ] Elegí filtros por SNR y rango de periodos de interés.
- [ ] Documenté frecuencia de corte, orden y fase.
- [ ] Apliqué taper y padding.
- [ ] Integré a velocidad y desplazamiento.
- [ ] Verifiqué drift final.
- [ ] Calculé PGA, PGV, PGD, Arias, duración, CAV.
- [ ] Calculé espectros.
- [ ] Probé sensibilidad del procesamiento.
- [ ] Para escalamiento, seleccioné registros físicamente compatibles.
- [ ] Calculé factores de escala y revisé que sean razonables.
- [ ] Si ajusté espectralmente, verifiqué que no distorsioné la señal.
- [ ] Comparé Arias y duración antes/después.
- [ ] Guardé una ficha reproducible del proceso.

---

## 34. Resumen ejecutivo

La corrección de acelerogramas es una combinación de física instrumental, estadística de ruido y criterio de ingeniería. La operación básica es restar una línea base $b(t)$, filtrar frecuencias no confiables e integrar con controles estrictos. Los polinomios de corrección se manipulan como modelos de sesgo de aceleración o como derivados de tendencias ajustadas en velocidad. Sus coeficientes se eligen por media pre-evento, mínimos cuadrados, restricciones de velocidad/desplazamiento final o sistemas KKT con restricciones físicas.

El filtrado debe diseñarse desde la relación señal/ruido y el rango de periodos del problema, no desde una gráfica “bonita” de desplazamiento. La verificación se realiza en aceleración, velocidad, desplazamiento, Fourier, espectros y parámetros energéticos.

El escalamiento sísmico empieza con selección física del registro. El escalamiento lineal conserva forma temporal y modifica amplitud; el ajuste espectral modifica la señal para calzar un objetivo y exige controles adicionales de drift, energía, duración y realismo. Para análisis no lineal y geotécnico, Arias, duración, PGV, PGD y pulsos near-fault son tan importantes como el espectro elástico.
