import kfp
from kfp import dsl


def test_for_test():
  # 假设 python:alpine3.6 就是我们要工作的镜像和执行的具体代码的地方
  # 通过 dsl.ContainerOp() 就把上述工作内容作为一个 component
  return dsl.ContainerOp(
      name='testfortest',
      image='python:alpine3.6',
      command=['sh', '-c'],
      arguments=["echo 'testfortest'"]
  )

# 然后就是设计 pipeline
# 本例很简单，只有一个 component
@dsl.pipeline(
    name='test for test',
    description='test for test'
)
def try_test_for_test():
  test_for_test()


if __name__ == '__main__':
  # 通过这个方法，生成一个 zip 文件，用来在 Pipeline UI 上上传的
  kfp.compiler.Compiler().compile(try_test_for_test, __file__ + '.zip')