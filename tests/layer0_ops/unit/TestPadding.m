classdef TestPadding < matlab.unittest.TestCase
    % TestPadding: unit tests for postpad, middlepad, fir2long, long2fir

    methods(TestClassSetup)
        function setupTestClass(testCase)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), ...
                                 '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    methods(Test)

        function testPostpadTargetLength(testCase)
            % postpad to target length L: output has length L
            rng(42);
            for trial = 1:10
                x = randn(100, 1);
                L = 200;
                result = postpad(x, L);
                testCase.verifyEqual(length(result), L);
            end
        end

        function testPostpadTruncation(testCase)
            % postpad truncation: output equals input(1:L) when L < length(input)
            rng(42);
            for trial = 1:10
                x = randn(200, 1);
                L = 100;
                result = postpad(x, L);
                testCase.verifyEqual(result, x(1:L), 'AbsTol', 1e-14);
            end
        end

        function testPostpadExtension(testCase)
            % postpad extension: first length(input) elements match input, rest are zero
            rng(42);
            for trial = 1:10
                x = randn(100, 1);
                L = 200;
                result = postpad(x, L);
                testCase.verifyEqual(result(1:length(x)), x, 'AbsTol', 1e-14);
                testCase.verifyEqual(result(length(x)+1:end), zeros(L-length(x), 1), 'AbsTol', 1e-14);
            end
        end

        function testMiddlepadIdentity(testCase)
            % middlepad(x, length(x)) = x (identity)
            rng(42);
            for trial = 1:10
                x = randn(128, 1);
                result = middlepad(x, length(x));
                testCase.verifyEqual(result, x, 'AbsTol', 1e-14);
            end
        end

        function testMiddlepadLength(testCase)
            % middlepad increases length correctly
            rng(42);
            for trial = 1:10
                x = randn(100, 1);
                L = 200;
                result = middlepad(x, L);
                testCase.verifyEqual(length(result), L);
            end
        end

        function testMiddlepadSymmetry(testCase)
            % middlepad preserves whole-point even (WPE) symmetry.
            % A WPE signal of even length N satisfies x(k) = x(N+2-k) for k=2..N,
            % equivalently result(2:end) == flipud(result(2:end)).
            %
            % Construction of WPE signal of even length 100:
            %   x = [a_half; flipud(a_half(2:end-1))] where a_half has 51 elements.
            %   This gives x(k) = x(102-k) for k=2..100  (genuine WPE, NOT palindrome).
            rng(42);
            for trial = 1:10
                % Build a genuinely WPE signal of even length 100
                a_half = randn(51, 1);
                x = [a_half; flipud(a_half(2:end-1))];  % length 51+49=100, WPE
                L = 300;
                result = middlepad(x, L);  % should preserve WPE

                % WPE check: result(k) = result(L+2-k) for k=2..L
                testCase.verifyEqual(result(2:end), flipud(result(2:end)), 'AbsTol', 1e-12, ...
                    sprintf('Trial %d: middlepad did not preserve WPE symmetry', trial));
            end
        end

        function testFir2LongLength(testCase)
            % fir2long output has correct length
            rng(42);
            for trial = 1:10
                h = randn(64, 1);
                L = 512;
                result = fir2long(h, L);
                testCase.verifyEqual(length(result), L);
            end
        end

        function testLong2FirLength(testCase)
            % long2fir output has correct length
            rng(42);
            for trial = 1:10
                H = randn(512, 1);
                h_len = 64;
                result = long2fir(H, h_len);
                testCase.verifyEqual(length(result), h_len);
            end
        end

        function testFir2LongLong2FirRoundtrip(testCase)
            % fir2long then long2fir roundtrip: long2fir(fir2long(h, 512), length(h)) ≈ h
            rng(42);
            for trial = 1:10
                h = randn(64, 1);
                L = 512;
                long_h = fir2long(h, L);
                recovered_h = long2fir(long_h, length(h));
                testCase.verifyEqual(recovered_h, h, 'AbsTol', 1e-12, ...
                    sprintf('Trial %d: FIR2LONG->LONG2FIR roundtrip failed', trial));
            end
        end

    end

end
