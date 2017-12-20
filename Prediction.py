import tensorflow as tf
import numpy


def x_norm(x):  # x값을 Normalization (이유: 학습을 빠르게 하고 학습과정 중 발산(양 또는 음의 무한대)을 피하기 위함)
    x = numpy.array(x)  # 리스트형태, 스칼라형태 관계없이 처리하기 위해 변환
    x = x / 100 * 0.99 + 0.01
    return x


# (요소값 - 최소값) / (최대값 - 최소값) * 0.99 + 0.01
# 최소값 0, 최대값 100 가정


# 섭씨와 화씨를 관찰하여 입력한다.
# 12.0 -> 53.6
# 28.0 -> 82.4
# 36.5 -> 97.7-2.76897920e+08
# 42.0 -> 107.6
# 29.8 -> 85.64
# 섭씨를 입력하면 화씨로 변환하고자 한다.
# 답을 살짝 알려주면 F = 1.8 * C + 32 이다.

x = [12.0, 28.0, 36.5, 42.0, 29.8]  # 섭씨(C) 입력
x = x_norm(x)

y = [53.6, 82.4, 97.7, 107.6, 85.64]  # 화씨(F) 출력(=Target)

# random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
# minval ~ maxval 사이의 숫자를 균등분포로 랜덤하게 생성한다.
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="Weight")  # 가중치(Weight) 변수
print(W)
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name="Bias")  # 편향(Bias) 변수
print(b)

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

hypothesis = tf.add(tf.multiply(W, X), b)  # 여기서 입력 값 1개와 가중치 1개씩 트레이닝에 이용되므로 요소간 곱셉 이용
# hypothesis = W * X + b   # 위 표현식과 정확히 같다.
# tf.multiply는 요소간 곱셉이다.
# tf.matmul는 행렬 곱셈이다. 즉 내적(dot product)으로 그 차이점을 잘 알아두어야 한다.
# tf.matmul로 바꾸면 "Shape must be rank 2" 오류가 발생한다. 실제 W는 [1]으로 rank 1이므로 사용하지 못한다.
# 2개 이상의 Feature를 이용한 경우부터 사용가능하다.
print(hypothesis)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3)  # 경사하강법(Gradient Descent) 사용
cost = tf.reduce_mean(tf.square(Y - hypothesis))  # 실제값에서 가정값의 차이에 제곱한 값들의 평균값을 비용으로 정의
train_op = optimizer.minimize(cost)  # 비용(=오류총합)를 최소화하도록 최적화

with tf.Session() as sess:  # 세션 블록 생성
    sess.run(tf.global_variables_initializer())

    print(sess.run(W), sess.run(b))

    for step in range(1000):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x, Y: y})

        print("Step: ", step, "  Cost: ", cost_val, "  W: ", sess.run(W), "  b: ", sess.run(b))

    print("X: 20, Y:", sess.run(hypothesis, feed_dict={X: x_norm(20)}))
    print("X: 30, Y:", sess.run(hypothesis, feed_dict={X: x_norm(30)}))
    print("X: 40, Y:", sess.run(hypothesis, feed_dict={X: x_norm(40)}))
    print("X: 50, Y:", sess.run(hypothesis, feed_dict={X: x_norm(50)}))
    print("X: 60, Y:", sess.run(hypothesis, feed_dict={X: x_norm(60)}))
