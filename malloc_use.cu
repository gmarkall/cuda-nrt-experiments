__global__ void use_malloc(int **p)
{
  *p = static_cast<int*>(malloc(sizeof(int)));
}

__global__ void use_free(int *p)
{
  free(p);
}
