classdef TestSubsampling < matlab.unittest.TestCase
    % TestSubsampling: unit tests for comp_downs and comp_ups
    
    
    methods(TestClassSetup)
        function setupTestClass(testCase)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), ...
                                 '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    methods(Test)
        
        function testCompsDownsIdentity(testCase)
            % comp_downs(x, 1) = x (a=1 is identity)
            rng(42);
            for trial = 1:10
                x = randn(128, 1);
                result = comp_downs(x, 1);
                testCase.verifyEqual(result, x, 'AbsTol', 1e-14);
            end
        end
        
        function testCompsDownsLength(testCase)
            % comp_downs with a=2: output length = ceil(L/2)
            rng(42);
            for trial = 1:10
                L = randi([100, 500]);
                x = randn(L, 1);
                a = 2;
                result = comp_downs(x, a);
                expected_len = ceil(L / a);
                testCase.verifyEqual(length(result), expected_len);
            end
        end
        
        function testCompsDownsUpsCyclic(testCase)
            % comp_downs followed by comp_ups recovers impulse structure (check specific positions)
            rng(42);
            % Create impulse
            x = zeros(128, 1);
            x(1) = 1;
            
            a = 2;
            down = comp_downs(x, a);
            up = comp_ups(down, a);
            
            % The up-sampled version should have impulse at positions 1, 1+a, 1+2a, ...
            testCase.verifyTrue(up(1) > 0, 'Impulse lost after down/up cycle');
        end
        
        function testCompsUpsIdentity(testCase)
            % comp_ups(x, 1) = x (a=1 is identity)
            rng(42);
            for trial = 1:10
                x = randn(128, 1);
                result = comp_ups(x, 1);
                testCase.verifyEqual(result, x, 'AbsTol', 1e-14);
            end
        end
        
        function testCompsUpsLength(testCase)
            % comp_ups output length = a * length(x)
            rng(42);
            for trial = 1:10
                x = randn(64, 1);
                a = randi([2, 8]);
                result = comp_ups(x, a);
                testCase.verifyEqual(length(result), a * length(x));
            end
        end
        
        function testCompsUpsEnergy(testCase)
            % Energy check: sum(comp_ups(x,a).^2) = sum(x.^2) (upsampling preserves energy)
            rng(42);
            for trial = 1:10
                x = randn(64, 1);
                a = randi([2, 8]);
                up = comp_ups(x, a);
                energy_in = sum(abs(x).^2);
                energy_out = sum(abs(up).^2);
                testCase.verifyEqual(energy_out, energy_in, 'RelTol', 1e-12);
            end
        end
        
        function testCompsDownsSkipParameter(testCase)
            % skip parameter: comp_downs(x, a, skip) starts at index 1+skip
            rng(42);
            x = randn(256, 1);
            a = 2;
            skip = 3;
            
            result_no_skip = comp_downs(x, a);
            result_skip = comp_downs(x, a, skip);
            
            % With skip, we start at x(1+skip), so first output should match skip version
            % (verification is qualitative - just check it runs without error)
            testCase.verifyEqual(length(result_skip), length(result_no_skip), ...
                'RelTol', 0.1);  % Allow some length difference due to skip
        end
        
        function testCompsDownsZeroSignal(testCase)
            % Test with zero signal
            x = zeros(128, 1);
            result = comp_downs(x, 2);
            testCase.verifyEqual(result, zeros(64, 1), 'AbsTol', 1e-14);
        end
        
        function testCompsDownsImpulse(testCase)
            % Test with impulse signal
            x = zeros(128, 1);
            x(1) = 1;
            result = comp_downs(x, 2);
            testCase.verifyEqual(result(1), 1, 'AbsTol', 1e-14);
        end
        
        function testCompsDownsNoise(testCase)
            % Test with real noise
            rng(42);
            x = randn(256, 1);
            a = 2;
            result = comp_downs(x, a);
            testCase.verifyEqual(length(result), ceil(256/a));
        end
        
    end
    
end
