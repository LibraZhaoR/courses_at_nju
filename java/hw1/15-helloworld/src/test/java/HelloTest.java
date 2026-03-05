import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import static org.junit.Assert.assertEquals;

public class HelloTest {

    @Test
    public void testHelloOutput() {
        PrintStream originalOut = System.out;
        ByteArrayOutputStream outContent = new ByteArrayOutputStream();
        System.setOut(new PrintStream(outContent));

        try {
            Hello.main(new String[]{});
            String output = outContent.toString().trim();
            assertEquals("Hello World!", output);
        } finally {
            System.setOut(originalOut);
        }
    }
}
