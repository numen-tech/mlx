// Copyright © 2023-2024 Apple Inc.
#include <memory>
#include <sstream>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

void new_stream(Stream stream) {
  if (stream.device == mlx::core::Device::gpu) {
    metal::device(stream.device).new_queue(stream.index);
  }
}

void eval(array& arr) {
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& d = metal::device(s.device);
  auto command_buffer = d.get_command_buffer(s.index);

  auto outputs = arr.outputs();
  {
    // If the array is a tracer hold a reference
    // to its inputs so they don't get donated
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }

    debug_set_primitive_buffer_label(command_buffer, arr.primitive());
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }
  std::unordered_set<std::shared_ptr<array::Data>> buffers;
  for (auto& in : arr.inputs()) {
    buffers.insert(in.data_shared_ptr());
  }
  for (auto& s : arr.siblings()) {
    buffers.insert(s.data_shared_ptr());
  }
  // Remove the output if it was donated to by an input
  if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end()) {
    buffers.erase(it);
  }

  if (d.command_buffer_needs_commit(s.index)) {
    d.end_encoding(s.index);
    scheduler::notify_new_task(s);
    // Errors are latched by the handler commit_command_buffer installs; a
    // throw here would run on Metal's completion thread and terminate.
    command_buffer->addCompletedHandler(
        [s, buffers = std::move(buffers)](MTL::CommandBuffer*) {
          scheduler::notify_task_completion(s);
        });
    d.commit_command_buffer(s.index);
    d.get_command_buffer(s.index);
  } else {
    command_buffer->addCompletedHandler(
        [buffers = std::move(buffers)](MTL::CommandBuffer*) {});
  }
}

void finalize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& d = metal::device(s.device);
  d.get_command_buffer(s.index);
  d.end_encoding(s.index);
  d.commit_command_buffer(s.index);
  d.get_command_buffer(s.index);
}

void synchronize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& d = metal::device(s.device);
  auto cb = d.get_command_buffer(s.index);
  cb->retain();
  d.end_encoding(s.index);
  d.commit_command_buffer(s.index);
  cb->waitUntilCompleted();
  auto error = d.take_error(s.index);
  if (error.empty() && cb->status() == MTL::CommandBufferStatusError) {
    std::ostringstream msg;
    msg << "[METAL] Command buffer execution failed: "
        << cb->error()->localizedDescription()->utf8String();
    error = msg.str();
  }
  cb->release();
  if (!error.empty()) {
    throw std::runtime_error(error);
  }
}

} // namespace mlx::core::gpu
