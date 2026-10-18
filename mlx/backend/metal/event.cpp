// Copyright © 2024 Apple Inc.

#include "mlx/event.h"
#include "mlx/backend/metal/device.h"
#include "mlx/scheduler.h"

namespace mlx::core {

Event::Event(Stream stream) : stream_(stream) {
  auto dtor = [](void* ptr) {
    auto p = metal::new_scoped_memory_pool();
    static_cast<MTL::SharedEvent*>(ptr)->release();
  };
  auto p = metal::new_scoped_memory_pool();
  event_ = std::shared_ptr<void>(
      metal::device(Device::gpu).mtl_device()->newSharedEvent(), dtor);
  if (event_ == nullptr) {
    throw std::runtime_error(
        "[Event::Event] Failed to create Metal shared event.");
  }
}

void Event::wait_for_signal() {
  static_cast<MTL::SharedEvent*>(event_.get())
      ->waitUntilSignaledValue(value(), -1); // never times out
}

void Event::wait() {
  wait_for_signal();
  // A failed command buffer signals the event itself and latches its error on
  // the stream; surface it here, on the waiting thread, where it can be caught.
  if (stream_.device == Device::gpu) {
    auto error = metal::device(stream_.device).take_error(stream_.index);
    if (!error.empty()) {
      throw std::runtime_error(error);
    }
  }
}

void Event::wait(Stream stream) {
  if (stream.device == Device::cpu) {
    // Scheduler workers run tasks with no catch frame, so this must not throw;
    // a latched error stays put for the caller's next GPU wait or synchronize.
    scheduler::enqueue(stream, [*this]() mutable { wait_for_signal(); });
  } else {
    auto& d = metal::device(stream.device);
    d.end_encoding(stream.index);
    auto command_buffer = d.get_command_buffer(stream.index);
    command_buffer->encodeWait(static_cast<MTL::Event*>(event_.get()), value());
    command_buffer->addCompletedHandler([*this](MTL::CommandBuffer*) {});
  }
}

void Event::signal(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [*this]() mutable {
      static_cast<MTL::SharedEvent*>(event_.get())->setSignaledValue(value());
    });
  } else {
    auto& d = metal::device(stream.device);
    d.end_encoding(stream.index);
    auto command_buffer = d.get_command_buffer(stream.index);
    command_buffer->encodeSignalEvent(
        static_cast<MTL::Event*>(event_.get()), value());
    command_buffer->addCompletedHandler([*this](MTL::CommandBuffer*) {});
    d.add_pending_signal(
        stream.index,
        event_,
        static_cast<MTL::SharedEvent*>(event_.get()),
        value());
  }
}

bool Event::is_signaled() const {
  return static_cast<MTL::SharedEvent*>(event_.get())->signaledValue() >=
      value();
}

} // namespace mlx::core
