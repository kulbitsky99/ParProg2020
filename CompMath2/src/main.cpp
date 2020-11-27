#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <unistd.h>
#include <cmath>

void calc(double* frame, uint32_t ySize, uint32_t xSize, double delta, int rank, int size)
{
    MPI_Bcast(&ySize, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&xSize, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    double* local_frame;
    double* new_frame;

    int* send_size, *send_shift, *recv_size, *recv_shift;
    if (rank == 0 && size > 0) 
    {
        send_size = new int[size];
        send_shift = new int[size];
        recv_size = new int[size];
        recv_shift = new int[size];
        uint32_t block_size = (ySize - 2 + size) / size;
        for (int i = 0; i < size; i++) 
        {
            int start_s = i * block_size - 1;
            uint32_t end_s = i * block_size + block_size + 1;
            if(start_s < 0)
              start_s = 0;
            if(end_s > ySize)
              end_s = ySize;
            
            send_size[i] = (end_s - start_s) * xSize;
            send_shift[i] = start_s * xSize;

            int start_r = i * block_size;
            uint32_t end_r = i * block_size + block_size;
            if(start_r == 0)
              start_r = 1;
            if(end_r > ySize - 1)
              end_r = ySize - 1;

            recv_size[i] = (end_r - start_r) * xSize;
            recv_shift[i] = start_r * xSize;
        }
        local_frame = new double[send_size[0]];
        new_frame = new double[xSize * ySize];
        double* temp_frame = new double[send_size[0]];
        uint32_t local_frame_size = send_size[0] / xSize;

        double diff = 0;
        do 
        {
            MPI_Scatterv(frame, send_size, send_shift, MPI_DOUBLE, local_frame, send_size[0], MPI_DOUBLE, 0, MPI_COMM_WORLD);
            for (uint32_t y = 0; y < local_frame_size; y++)
            {
                temp_frame[y * xSize] = local_frame[y * xSize];
                temp_frame[y * xSize + xSize - 1] = local_frame[y * xSize + xSize - 1];
            }
            for (uint32_t x = 1; x < xSize - 1; x++)
            {
                temp_frame[x] = local_frame[x];
                temp_frame[(local_frame_size - 1) * xSize + x] = local_frame[(local_frame_size - 1) * xSize + x];
            }
            for (uint32_t y = 1; y < local_frame_size - 1; y++)
            {
                for (uint32_t x = 1; x < xSize - 1; x++)
                {
                    int index = y * xSize + x;
                    temp_frame[index] = (local_frame[(y + 1) * xSize + x] + local_frame[(y - 1) * xSize + x] + local_frame[index + 1] + local_frame[index - 1]) / 4.;
                }
            }
            MPI_Gatherv(temp_frame + xSize, recv_size[0], MPI_DOUBLE, new_frame, recv_size, recv_shift, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            diff = 0;
            for (uint32_t y = 1; y < ySize - 1; y++)
                for (uint32_t x = 1; x < xSize - 1; x++)
                {
                    int index = y * xSize + x;
                    diff += std::abs(new_frame[index] - frame[index]);
                    frame[index] = new_frame[index];
                }
            int flag = 0;
            if (diff < delta)
                flag = 1;
            MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        } while (diff > delta);

        delete [] new_frame;
        delete [] local_frame;
        delete [] temp_frame;
        delete [] send_size;
        delete [] recv_size;
        delete [] send_shift;
        delete [] recv_shift;
    } 
    else
    {
        uint32_t block_size = (ySize - 2 + size) / size;
        int start_s = rank * block_size - 1;
        uint32_t end_s = rank * block_size + block_size + 1;
        if(start_s < 0)
          start_s = 0;
        if(end_s > ySize)
          end_s = ySize;

        uint32_t to_recv = (end_s - start_s) * xSize;
        uint32_t local_frame_size = end_s - start_s;

        int start_r = rank * block_size;
        uint32_t end_r = rank * block_size + block_size;
        if(end_r > ySize - 1)
          end_r = ySize - 1;
        int32_t to_send = (end_r - start_r) * xSize;
        
        local_frame = new double[to_recv];
        double* temp_frame = new double[to_recv];
        int flag = 0;

        do 
        {
            MPI_Scatterv(frame, send_size, send_shift, MPI_DOUBLE, local_frame, to_recv, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            for (uint32_t y = 0; y < local_frame_size; y++) 
            {
                temp_frame[y * xSize] = local_frame[y * xSize];
                temp_frame[y * xSize + xSize - 1] = local_frame[y * xSize + xSize - 1];
            }
            for (uint32_t x = 1; x < xSize - 1; x++)
            {
                temp_frame[x] = local_frame[x];
                temp_frame[(local_frame_size - 1) * xSize + x] = local_frame[(local_frame_size - 1) * xSize + x];
            }
            for (uint32_t y = 1; y < local_frame_size - 1; y++)
            {
                for (uint32_t x = 1; x < xSize - 1; x++)
                {
                    int index = y * xSize + x;
                    temp_frame[index] = (local_frame[(y + 1) * xSize + x] + local_frame[(y - 1) * xSize + x] + local_frame[index + 1] + local_frame[index - 1]) / 4.;
                }
            }
            std::swap(temp_frame, local_frame);
            MPI_Gatherv(local_frame + xSize, to_send, MPI_DOUBLE, new_frame, recv_size, recv_shift, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        } while (!flag);
        delete [] local_frame;
        delete [] temp_frame;
    }
}

int main(int argc, char** argv)
{
  int rank = 0, size = 0, status = 0;
  double delta = 0;
  uint32_t ySize = 0, xSize = 0;
  double* frame = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
  {
    // Check arguments
    if (argc != 3)
    {
      std::cout << "[Error] Usage <inputfile> <output file>\n";
      status = 1;
      MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Prepare input file
    std::ifstream input(argv[1]);
    if (!input.is_open())
    {
      std::cout << "[Error] Can't open " << argv[1] << " for write\n";
      status = 1;
      MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Read arguments from input
    input >> ySize >> xSize >> delta;
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    frame = new double[ySize * xSize];

    for (uint32_t y = 0; y < ySize; y++)
    {
     for (uint32_t x = 0; x < xSize; x++)
      {
        input >> frame[y*xSize + x];
      }
    }
    input.close();
  } else {
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != 0)
    {
      return 1;
    }
  }

  calc(frame, ySize, xSize, delta, rank, size);

  if (rank == 0)
  {
    // Prepare output file
    std::ofstream output(argv[2]);
    if (!output.is_open())
    {
      std::cout << "[Error] Can't open " << argv[2] << " for read\n";
      delete frame;
      return 1;
    }
    for (uint32_t y = 0; y < ySize; y++)
    {
      for (uint32_t x = 0; x < xSize; x++)
      {
        output << " " << frame[y*xSize + x];
      }
      output << std::endl;
    }
    output.close();
    delete frame;
  }

  MPI_Finalize();
  return 0;
}