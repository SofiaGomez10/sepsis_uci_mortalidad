## Qué es un coeficiente en la regresión logística
claves: coeficiente, peso, beta, que significa, como se interpreta, numero de cada variable
El modelo suma puntos de riesgo. Cada variable tiene un peso fijo, llamado coeficiente, que se multiplica por el valor del paciente. Un coeficiente positivo significa que, a mayor valor de la variable, mayor riesgo estimado; uno negativo, que lo reduce; uno igual a cero, que el modelo no usa esa variable. La suma de todos esos productos es un puntaje (el log-odds) que luego se convierte en una probabilidad entre 0 y 100 %. Es la misma lógica de una escala como SOFA o APACHE II, con la diferencia de que aquí los pesos se aprendieron de los datos en lugar de fijarse por consenso. Los coeficientes se calculan sobre variables estandarizadas, por lo que permiten comparar la importancia relativa de variables medidas en unidades distintas.

## Qué es un odds ratio (OR)
claves: odds ratio, OR, razon de odds, razon de momios, momios, chance, como leer el OR, duplica
Los odds son la probabilidad de que ocurra el evento dividida entre la probabilidad de que no ocurra: un riesgo de 50 % equivale a odds de 1 (uno a uno) y un riesgo de 67 % a odds de 2 (dos a uno). El odds ratio es el factor por el que se multiplican los odds cuando la variable sube una unidad o cuando la condición pasa de ausente a presente. Un OR de 2 duplica los odds; un OR de 1 no cambia nada; un OR menor de 1 los reduce. Atención: duplicar los odds no es duplicar la probabilidad. Si el riesgo de partida es 42 %, un OR de 2 lo lleva a cerca de 59 %, no a 84 %. El OR se obtiene elevando el número e al coeficiente.

## Diferencia entre odds y probabilidad
claves: odds, probabilidad, diferencia, riesgo, porcentaje, log-odds, logit, puntaje
La probabilidad va de 0 a 100 % y es lo que se muestra en pantalla. Los odds son otra forma de expresar lo mismo (probabilidad dividida entre su complemento) y el log-odds es su logaritmo. El modelo trabaja internamente en log-odds porque en esa escala los efectos de las variables simplemente se suman. Al final se aplica la función sigmoide, que convierte el log-odds en probabilidad. Por eso un mismo factor no aporta siempre los mismos puntos porcentuales: su efecto depende del riesgo del que parte el paciente.

## Qué hace la regularización L1 y por qué hay coeficientes en cero
claves: L1, lasso, regularizacion, penalizacion, coeficiente cero, por que no usa, variable eliminada, MAP en cero, SOFA cero
La regularización L1 es una penalización que se aplica al entrenar: el modelo solo puede darle peso a una variable si eso mejora de verdad la predicción. El efecto práctico es que elimina, dejando en cero, las variables que repiten información que ya aportan otras, y encoge el resto hacia cero. Que una variable quede en cero no significa que carezca de importancia clínica: significa que, teniendo las demás, no agrega información nueva en estos datos. Otra consecuencia es que los odds ratios del modelo son conservadores, es decir, algo menores que los que daría un análisis sin penalización.

## Qué significa estandarizar una variable
claves: estandarizar, escalar, desviacion estandar, DE, z, unidades, por DE
Antes de entrar al modelo, a cada variable numérica se le resta el promedio de los pacientes de entrenamiento y se divide por su desviación estándar. Así, edad en años y lactato en mmol/L quedan en la misma escala y sus coeficientes se pueden comparar. Por eso se reportan dos odds ratios: por desviación estándar (para comparar variables entre sí) y por unidad (para la lectura clínica directa).

## Qué es el AUC o área bajo la curva ROC
claves: AUC, ROC, discriminacion, area bajo la curva, que tan bueno, capacidad de distinguir, 0.59, 0.6
El AUC mide la capacidad del modelo para ordenar a los pacientes por riesgo. Se interpreta así: si se toman al azar un paciente que falleció y uno que sobrevivió, el AUC es la probabilidad de que el modelo le haya asignado mayor riesgo al que falleció. Un AUC de 0,5 equivale a lanzar una moneda y 1,0 es separación perfecta. Como orientación general, por debajo de 0,7 la discriminación se considera pobre, entre 0,7 y 0,8 aceptable y por encima de 0,8 buena. El AUC no depende de ningún punto de corte.

## Qué es la calibración y el puntaje de Brier
claves: calibracion, calibrado, Brier, Platt, se puede creer el porcentaje, confiable la probabilidad, pendiente
Un modelo está calibrado cuando sus porcentajes coinciden con lo que ocurre: de cada 100 pacientes a los que les asigna 60 % de riesgo, deben fallecer cerca de 60. Es una propiedad distinta de la discriminación: un modelo puede ordenar bien y aun así exagerar o subestimar los riesgos. El puntaje de Brier resume el error de las probabilidades (0 es perfecto; menor es mejor). En este modelo las probabilidades se ajustaron con calibración de Platt, una corrección que no altera el orden de los pacientes y solo corrige la escala.

## Sensibilidad, especificidad, VPP y VPN
claves: sensibilidad, especificidad, valor predictivo positivo, negativo, VPP, VPN, falsos positivos, falsos negativos, recall, precision
La sensibilidad es la proporción de pacientes que fallecieron y que el modelo había señalado como de riesgo. La especificidad es la proporción de sobrevivientes que el modelo dejó correctamente por debajo del corte. El valor predictivo positivo (VPP) es la proporción de alertas que correspondieron a un fallecimiento, y el negativo (VPN), la proporción de pacientes no alertados que sobrevivieron. Todas dependen del punto de corte elegido: bajarlo aumenta la sensibilidad y las falsas alarmas; subirlo hace lo contrario.

## Qué es la contribución de cada variable (valores SHAP)
claves: contribucion, SHAP, explicacion, barras, por que este paciente, que variable pesa mas, puntos porcentuales, aporta
Para cada paciente, la aplicación muestra cuánto empuja cada variable el riesgo hacia arriba o hacia abajo respecto a un paciente promedio de la cohorte de entrenamiento. Se calcula como el coeficiente multiplicado por la diferencia entre el valor del paciente y el promedio. En un modelo lineal como este, ese cálculo es exacto y coincide con los valores SHAP. La suma de todas las contribuciones más el riesgo del paciente promedio reproduce exactamente la predicción. Es una explicación de cómo razona el modelo, no una afirmación de causa: que una variable aporte riesgo no implica que corregirla lo reduzca.

## Qué pasa cuando falta un dato
claves: dato faltante, sin dato, falta, vacio, obligatorio, imputacion, KNN, vecinos, no tengo el lactato, estimado, por que debo ingresar todos
Todos los campos son obligatorios: si falta alguno, la aplicación no calcula y señala cuál falta. El modelo no estima datos ausentes de un paciente real, porque el resultado dejaría de describir a ese paciente y se parecería al promedio de la cohorte. La única excepción es la presión arterial media: si se deja vacía, se calcula a partir de la sistólica y la diastólica con la fórmula habitual. Si un resultado de laboratorio aún no está disponible, lo indicado es esperar a tenerlo. La imputación por vecinos más cercanos (KNN) se usó únicamente para entrenar el modelo, porque el conjunto de datos traía cerca de 7 % de valores faltantes por variable y descartar esos pacientes habría reducido la muestra a menos de la mitad.

## Qué es la validación cruzada y el conjunto de prueba
claves: validacion cruzada, folds, conjunto de prueba, test, entrenamiento, sobreajuste, memorizo, generaliza
Los pacientes se dividieron una sola vez en 75 % para entrenar y 25 % para probar. Todas las decisiones (variables, hiperparámetros, calibración y puntos de corte) se tomaron solo con el grupo de entrenamiento, repartiéndolo en cinco partes y usando por turnos cuatro para ajustar y una para verificar; eso es la validación cruzada. El grupo de prueba se usó una única vez al final para medir el desempeño. Que el resultado en prueba sea similar al de validación cruzada indica que el modelo no memorizó los datos (no hay sobreajuste).

## Qué es la curva de decisión y el beneficio neto
claves: curva de decision, beneficio neto, utilidad clinica, sirve usarlo, vale la pena
El beneficio neto compara tres estrategias para decidir a quién intensificar la vigilancia: usar el modelo, intensificar a todos o no intensificar a nadie. Suma los pacientes de riesgo correctamente identificados y resta las falsas alarmas, ponderadas según qué tan costosa se considere una falsa alarma frente a un caso no detectado. Si la curva del modelo queda por encima de las otras dos en el rango de umbrales clínicamente razonable, usarlo aporta más que las estrategias simples.

## Por qué se usa class_weight balanced
claves: class weight, balanced, clases balanceadas, pesos de clase, desbalance
Al entrenar se dio el mismo peso total a fallecidos y sobrevivientes, aunque los fallecidos son algo menos (46,5 %). Esto evita que el modelo se incline hacia la clase más frecuente y mejora la detección de fallecidos. Su efecto secundario es que infla ligeramente las probabilidades, y por eso se añadió la calibración posterior.
