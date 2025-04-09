import vgamepad
import torch


class GamePad:
    def __init__(self):
        self.gamepad = vgamepad.VX360Gamepad()
        self.UP = vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP
        self.DOWN = vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN
        self.LEFT = vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT
        self.RIGHT = vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT

        self.START = vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_START
        self.BACK = vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_BACK
        self.GUIDE = vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_GUIDE

        self.LEFT_THUMB = vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB
        self.RIGHT_THUMB = vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB
        self.LEFT_SHOULDER = vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER
        self.RIGHT_SHOULDER = vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER

        self.A = vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_A
        self.B = vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_B
        self.X = vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_X
        self.Y = vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_Y

    def left_trigger(self, value):
        self.gamepad.left_trigger_float(value_float=value)
        # 左扳机轴 value改成0.0到1.0之间的浮点值

    def right_trigger(self, value):
        self.gamepad.right_trigger_float(value_float=value)
        # 右扳机轴 value改成0.0到1.0之间的浮点值

    def left_joystick(self, x_value, y_value):
        self.gamepad.left_joystick_float(x_value_float=x_value, y_value_float=y_value)
        # 左摇杆XY轴  x_values和y_values改成-1.0到1.0之间的浮点值

    def right_joystick(self, x_value, y_value):
        self.gamepad.right_joystick_float(x_value_float=x_value, y_value_float=y_value)
        # 右摇杆XY轴  x_values和y_values改成-1.0到1.0之间的浮点值

    def press_button(self, button):
        self.gamepad.press_button(button=button)

    def release_button(self, button):
        self.gamepad.release_button(button=button)

    def Tensor3Controll(self, Tensor3Controll_: torch.Tensor):
        Tcon3list = Tensor3Controll_.squeeze(dim=0).tolist()
        # print(Tcon3list)
        self.left_joystick(Tcon3list[0], Tcon3list[1])
        self.right_trigger(0.)
        self.left_trigger(0.)
        # LEFT_TRIGGER(1.0)
        print(Tcon3list)
        if (Tcon3list[2] <= 0):
            self.left_trigger(-Tcon3list[2])
            self.right_trigger(0.)
        else:
            self.right_trigger(Tcon3list[2])
            self.left_trigger(0.)

    def List3Controll(self, List3Controll_: list):
        # print(Tcon3list)
        self.left_joystick(List3Controll_[1], List3Controll_[0])
        self.right_trigger(0.)
        self.left_trigger(0.)
        # LEFT_TRIGGER(1.0)
        print(List3Controll_)
        if (List3Controll_[2] <= 0):
            self.left_trigger(-List3Controll_[2])
            self.right_trigger(0.)
        else:
            self.right_trigger(List3Controll_[2])
            self.left_trigger(0.)

    def del_gamepad(self):
        self.gamepad.reset()
        self.gamepad.update()
        del self.gamepad
